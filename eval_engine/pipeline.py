"""
Evaluation Pipeline
===================
LLM Response Decision Engine.

Architecture:
  Query → Context → LLM → Response
                               ↓
                         Scoring Layer
                         ├── Attribution  (is it grounded?)
                         ├── Specificity  (is it concrete?)
                         ├── Relevance    (does it answer the query?)
                         ├── Context quality (is retrieval good?)
                         └── Disagreement (are scorers aligned?)
                               ↓
                         Decision Layer
                         ├── 3D evaluation: grounding × specificity × agreement
                         ├── High specificity + low attribution = hallucination
                         └── ACCEPT | REVIEW | REJECT
                               ↓
                         Action Layer
                         serve | retry | retrieve_more | regenerate

Key insight:
  Most eval systems work in 1D (good vs bad).
  This system works in 3D:
    - Grounding  (truth)
    - Specificity (confidence)
    - Agreement  (certainty)
  That's exactly how humans evaluate answers.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from scorers import faithfulness, relevance, context
from eval_engine.aggregator import aggregate, AggregatedScore


# ── Decision thresholds ───────────────────────────────────────────────────────
REJECT_THRESHOLD         = 0.45
REVIEW_THRESHOLD         = 0.65

# Attribution bands
HALLUCINATION_HARD       = 0.35   # confirmed hallucination → always REJECT
HALLUCINATION_CONFIDENT  = 0.45   # low attribution + high specificity → REJECT
                                  # "confident but wrong" = hallucination
UNCERTAINTY_THRESHOLD    = 0.55   # weak grounding → REVIEW

# Other gates
RELEVANCE_MIN            = 0.30   # below this → off-topic
CONTEXT_QUALITY_MIN      = 0.40   # below this → retrieval is root cause
DISAGREEMENT_THRESHOLD   = 0.12   # above this → force REVIEW

# LLM judge fires only in the uncertain zone (cost-effective)
LLM_JUDGE_LOWER          = 0.45
LLM_JUDGE_UPPER          = 0.65

# Specificity threshold for "confident hallucination" detection
CONFIDENT_HALLUC_SPEC    = 0.60   # high specificity + low attribution = hallucination


def _disagreement(scores: list[float]) -> float:
    """Standard deviation of scorer outputs. High = system is uncertain."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return round(math.sqrt(sum((s - mean) ** 2 for s in scores) / n), 4)


def _confidence_pct(
    final_score: float,
    threshold: float,
    attribution: float,
    disagreement: float,
) -> int:
    """
    Meaningful confidence score combining three signals:

      margin      = how far the score is from the decision boundary
      attribution = how grounded the response is (truth signal)
      stability   = 1 - normalised disagreement (scorer agreement)

    Weights:
      50% margin + 30% attribution + 20% stability

    A strong score, well-grounded, with aligned scorers → high confidence.
    A borderline score with diverging scorers → low confidence.
    """
    margin     = min(1.0, abs(final_score - threshold) * 2)
    stability  = max(0.0, 1.0 - disagreement / DISAGREEMENT_THRESHOLD)
    raw = 0.50 * margin + 0.30 * attribution + 0.20 * stability
    return min(100, round(raw * 100))


def decision_layer(
    final_score: float,
    attribution: float,
    specificity: float,
    relevance_score: float,
    context_quality: float,
    hallucination_detected: bool,
    disagreement: float,
) -> tuple[str, str, str, str, str, int]:
    """
    3D decision logic: grounding × specificity × agreement.

    Returns (decision, reason, action, action_why, failure_type, confidence_pct)

    Failure taxonomy:
      hallucination    — fabricated claims (confirmed or confident)
      weak_grounding   — vague, not concrete
      off_topic        — doesn't answer the query
      poor_retrieval   — bad context is the root cause
      uncertain        — scorers disagree
      none             — all gates passed

    The critical rule:
      High specificity + low attribution = hallucination.
      A confident wrong answer is more dangerous than a vague one.
    """
    conf = _confidence_pct(final_score, REVIEW_THRESHOLD, attribution, disagreement)

    # ── 1. Low attribution — split by specificity ─────────────────────────────
    # Low attribution + high specificity = confident hallucination → REJECT
    # Low attribution + low specificity  = uncertain / vague      → REVIEW
    if attribution < HALLUCINATION_HARD:
        if specificity > 0.50:
            return (
                "REJECT",
                f"Confident hallucination — attribution={attribution:.3f} (low grounding), "
                f"specificity={specificity:.3f} (high confidence). "
                f"Response sounds authoritative but is fabricated.",
                "regenerate_with_grounding_prompt",
                "Confident but ungrounded response is more dangerous than a vague one",
                "hallucination",
                _confidence_pct(attribution, HALLUCINATION_HARD, attribution, disagreement),
            )
        else:
            return (
                "REVIEW",
                f"Uncertain / vague response — attribution={attribution:.3f} (low grounding), "
                f"specificity={specificity:.3f} (low confidence). "
                f"Response is weakly grounded and non-specific; not a confirmed hallucination.",
                "retry_with_specific_prompt",
                "Response is vague and uncertain; a better prompt may improve grounding",
                "weak_grounding",
                _confidence_pct(attribution, HALLUCINATION_HARD, attribution, disagreement),
            )

    # ── 2. Confident hallucination (low attribution + high specificity) ───────
    # The most dangerous failure: sounds authoritative, but is wrong.
    # High specificity + low attribution = hallucination, not weak grounding.
    if attribution < HALLUCINATION_CONFIDENT and specificity > CONFIDENT_HALLUC_SPEC:
        return (
            "REJECT",
            f"Confident hallucination detected — attribution={attribution:.3f} "
            f"(low grounding) but specificity={specificity:.3f} (high confidence). "
            f"Response sounds authoritative but is not grounded in context.",
            "regenerate_with_grounding_prompt",
            "Confident but ungrounded response is more dangerous than a vague one",
            "hallucination",
            _confidence_pct(attribution, HALLUCINATION_CONFIDENT, attribution, disagreement),
        )

    # ── 3. Poor retrieval (context is root cause) ─────────────────────────────
    if context_quality < CONTEXT_QUALITY_MIN and final_score < REVIEW_THRESHOLD:
        return (
            "REVIEW",
            f"Poor retrieval quality — context_quality={context_quality:.3f} "
            f"(threshold: {CONTEXT_QUALITY_MIN}). Model may lack sufficient context "
            f"to answer faithfully.",
            "retrieve_more_documents",
            "Root cause is retrieval, not the model — improve context before regenerating",
            "poor_retrieval",
            _confidence_pct(context_quality, CONTEXT_QUALITY_MIN, attribution, disagreement),
        )

    # ── 4. Hard guardrail: suspected hallucination + poor grounding → REJECT ──
    # "Hmm, maybe hallucinated… but let it pass" is how bad answers reach users.
    # If attribution AND context quality are both below threshold, reject.
    if attribution < 0.55 and context_quality < 0.50:
        return (
            "REJECT",
            f"Hallucination guardrail triggered — attribution={attribution:.3f} "
            f"and context_quality={context_quality:.3f} are both below safe thresholds. "
            f"Response cannot be trusted without better grounding.",
            "retrieve_more_documents",
            "Improve retrieval quality first, then regenerate with stronger context",
            "hallucination",
            _confidence_pct(attribution, UNCERTAINTY_THRESHOLD, attribution, disagreement),
        )

    # ── 4. Weak grounding (vague but not wrong) ───────────────────────────────
    if attribution < UNCERTAINTY_THRESHOLD:
        return (
            "REVIEW",
            f"Weak grounding — attribution={attribution:.3f}, "
            f"specificity={specificity:.3f} (threshold: {UNCERTAINTY_THRESHOLD}). "
            f"Response is vague but not a confirmed hallucination.",
            "retry_with_specific_prompt",
            "A more specific prompt may improve grounding and concreteness",
            "weak_grounding",
            _confidence_pct(attribution, UNCERTAINTY_THRESHOLD, attribution, disagreement),
        )

    # ── 5. Off-topic ──────────────────────────────────────────────────────────
    if relevance_score < RELEVANCE_MIN:
        return (
            "REVIEW",
            f"Off-topic response — relevance={relevance_score:.3f} "
            f"(threshold: {RELEVANCE_MIN}). Response does not address the query.",
            "retry_with_clearer_query",
            "Rephrase the query more explicitly before regenerating",
            "off_topic",
            _confidence_pct(relevance_score, RELEVANCE_MIN, attribution, disagreement),
        )

    # ── 6. High scorer disagreement (uncertain) ───────────────────────────────
    if disagreement > DISAGREEMENT_THRESHOLD:
        return (
            "REVIEW",
            f"High scorer disagreement — std={disagreement:.3f} "
            f"(threshold: {DISAGREEMENT_THRESHOLD}). Scoring signals conflict; "
            f"system cannot confidently accept or reject.",
            "optional_human_review",
            "Conflicting scorer signals; human judgment recommended",
            "uncertain",
            _confidence_pct(final_score, REVIEW_THRESHOLD, attribution, disagreement),
        )

    # ── 7. Borderline quality ─────────────────────────────────────────────────
    if final_score < REVIEW_THRESHOLD:
        return (
            "REVIEW",
            f"Borderline quality — final={final_score:.3f} "
            f"(auto-accept threshold: {REVIEW_THRESHOLD}).",
            "optional_human_review",
            "Score is acceptable but not high enough for automatic serving",
            "none",
            conf,
        )

    # ── 8. All gates passed ───────────────────────────────────────────────────
    return (
        "ACCEPT",
        f"All quality gates passed — final={final_score:.3f}, "
        f"attribution={attribution:.3f}, disagreement={disagreement:.3f}.",
        "serve_response",
        "Response is grounded, relevant, and high quality",
        "none",
        conf,
    )


@dataclass
class EvalResult:
    """Full evaluation result from EvalPipeline.evaluate()"""
    query: str
    response: str

    # scores
    faithfulness_score: float
    attribution_score: float
    specificity_score: float
    relevance_score: float
    context_quality_score: float
    consistency_score: Optional[float]
    final_score: float
    passed: bool
    disagreement: float

    # decision
    decision: str
    decision_reason: str
    action: str
    action_why: str
    failure_type: str
    confidence_pct: int
    hallucination_status: str   # none | suspected | confirmed

    # flags
    needs_llm_review: bool
    used_llm_judge: bool

    # diagnostics
    low_confidence_sentences: list = field(default_factory=list)
    llm_judge_scores: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def __str__(self) -> str:
        icons   = {"ACCEPT": "✅", "REVIEW": "🔍", "REJECT": "🚫"}
        hall_labels = {
            "none":      "✓  No hallucination",
            "suspected": "⚠️  Suspected weak grounding",
            "confirmed": "🚫 Hallucination confirmed",
        }
        d_icon = icons.get(self.decision, "?")
        hall   = hall_labels.get(self.hallucination_status, self.hallucination_status)
        lines  = [
            f"\n{'─'*56}",
            f"  LLM Eval Result  {'✅ PASSED' if self.passed else '❌ FAILED'}",
            f"{'─'*56}",
            f"  Final Score       : {self.final_score:.3f}",
            f"  Attribution       : {self.attribution_score:.3f}   (grounding)",
            f"  Specificity       : {self.specificity_score:.3f}   (concreteness)",
            f"  Relevance         : {self.relevance_score:.3f}",
            f"  Context Quality   : {self.context_quality_score:.3f}",
            f"  Disagreement      : {self.disagreement:.3f}   (scorer std dev)",
        ]
        if self.consistency_score is not None:
            lines.append(f"  Consistency       : {self.consistency_score:.3f}")
        lines.append(f"  {hall}")
        if self.failure_type != "none":
            lines.append(f"  Failure Type      : {self.failure_type}")
        if self.disagreement > DISAGREEMENT_THRESHOLD:
            lines.append(f"  ⚠️  High scorer disagreement: {self.disagreement:.3f}")
        lines += [
            f"  Decision          : {d_icon} {self.decision}  (confidence: {self.confidence_pct}%)",
            f"  Reason            : {self.decision_reason}",
            f"  Next Action       : {self.action}",
            f"  Why               : {self.action_why}",
            f"  LLM Judge Used    : {'Yes' if self.used_llm_judge else 'No (heuristics sufficient)'}",
            f"  Latency           : {self.latency_ms:.1f}ms",
            f"{'─'*56}",
        ]
        if self.low_confidence_sentences:
            lines.append("  Low-confidence sentences:")
            for s in self.low_confidence_sentences:
                lines.append(f"    • {s[:80]}{'...' if len(s) > 80 else ''}")
            lines.append(f"{'─'*56}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable result — for logging, APIs, and dashboards."""
        return {
            "decision": self.decision,
            "confidence_pct": self.confidence_pct,
            "failure_type": self.failure_type,
            "hallucination_status": self.hallucination_status,
            "next_action": self.action,
            "action_why": self.action_why,
            "scores": {
                "final": self.final_score,
                "attribution": self.attribution_score,
                "specificity": self.specificity_score,
                "relevance": self.relevance_score,
                "context_quality": self.context_quality_score,
                "disagreement": self.disagreement,
            },
            "explanations": {
                "reason": self.decision_reason,
                "low_confidence_sentences": self.low_confidence_sentences,
            },
            "meta": {
                "passed": self.passed,
                "used_llm_judge": self.used_llm_judge,
                "latency_ms": self.latency_ms,
            },
        }


class EvalPipeline:
    """
    LLM Response Decision Engine: Score → Decide → Act.

    Parameters
    ----------
    use_llm_judge : bool
        Enables LLM-as-judge for edge cases (score in 0.45–0.65 zone only).
        Default: False. Zero API cost in heuristic-only mode.
    llm_fn : Callable[[str], str], optional
        LLM callable for consistency scoring.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_fn: Optional[Callable[[str], str]] = None,
        api_key: Optional[str] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.llm_fn = llm_fn
        self.api_key = api_key

    def evaluate(
        self,
        query: str,
        context_text: str,
        response: str,
        run_consistency: bool = False,
    ) -> EvalResult:
        t0 = time.perf_counter()

        # ── Step 1: Score ─────────────────────────────────────────────────────
        faith_result = faithfulness.score(context_text, response)
        rel_result   = relevance.score(query, response)
        ctx_result   = context.score(context_text, response)

        # ── Step 2: Consistency (optional) ───────────────────────────────────
        consistency_score: Optional[float] = None
        if run_consistency and self.llm_fn is not None:
            from scorers.consistency import score as cscore
            consistency_score = cscore(query, response, self.llm_fn).score

        # ── Step 3: Aggregate ─────────────────────────────────────────────────
        agg: AggregatedScore = aggregate(
            faith_result, rel_result, ctx_result, consistency_score
        )

        # ── Step 4: Disagreement signal ───────────────────────────────────────
        disagree = _disagreement([
            faith_result.score,
            rel_result.score,
            ctx_result.score,
        ])

        # ── Step 5: LLM judge — edge cases only ───────────────────────────────
        used_llm   = False
        llm_scores: dict = {}
        in_edge    = LLM_JUDGE_LOWER < agg.final_score < LLM_JUDGE_UPPER

        if self.use_llm_judge and in_edge:
            used_llm = True
            try:
                from llm_judge.judge import judge_faithfulness, judge_relevance
                lf = judge_faithfulness(context_text, response, self.api_key)
                lr = judge_relevance(query, response, self.api_key)
                faith_result.score            = round(0.60 * faith_result.score + 0.40 * lf.score, 4)
                faith_result.attribution_score = faith_result.score
                rel_result.score              = round(0.60 * rel_result.score   + 0.40 * lr.score, 4)
                llm_scores = {
                    "llm_faithfulness": round(lf.score, 4),
                    "llm_relevance":    round(lr.score, 4),
                    "llm_faith_reason": lf.reason,
                    "llm_rel_reason":   lr.reason,
                }
                agg     = aggregate(faith_result, rel_result, ctx_result, consistency_score)
                disagree = _disagreement([faith_result.score, rel_result.score, ctx_result.score])
            except Exception as e:
                llm_scores = {"llm_error": str(e)}

        # ── Step 6: Hallucination status ──────────────────────────────────────
        attr = faith_result.attribution_score
        spec = faith_result.specificity_score
        hall = faith_result.hallucination_detected

        if attr < HALLUCINATION_HARD and spec > 0.50 and hall:
            hallucination_status = "confirmed"
        elif (attr < HALLUCINATION_CONFIDENT and spec > CONFIDENT_HALLUC_SPEC) or \
             (attr < UNCERTAINTY_THRESHOLD and hall):
            hallucination_status = "suspected"
        else:
            hallucination_status = "none"

        # ── Step 7: Decision ──────────────────────────────────────────────────
        decision, reason, action, action_why, failure_type, confidence_pct = decision_layer(
            final_score=agg.final_score,
            attribution=attr,
            specificity=spec,
            relevance_score=rel_result.score,
            context_quality=ctx_result.score,
            hallucination_detected=hall,
            disagreement=disagree,
        )

        latency = (time.perf_counter() - t0) * 1000

        return EvalResult(
            query=query,
            response=response,
            faithfulness_score=faith_result.score,
            attribution_score=attr,
            specificity_score=spec,
            relevance_score=agg.relevance,
            context_quality_score=agg.context_quality,
            consistency_score=agg.consistency,
            final_score=agg.final_score,
            passed=agg.passed,
            disagreement=disagree,
            decision=decision,
            decision_reason=reason,
            action=action,
            action_why=action_why,
            failure_type=failure_type,
            confidence_pct=confidence_pct,
            hallucination_status=hallucination_status,
            needs_llm_review=agg.needs_llm_review,
            used_llm_judge=used_llm,
            low_confidence_sentences=faith_result.low_confidence_sentences,
            llm_judge_scores=llm_scores,
            latency_ms=round(latency, 2),
        )
