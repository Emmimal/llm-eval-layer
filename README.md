# llm-eval-layer

A pure-Python decision engine for LLM responses — faithfulness scoring, hallucination
detection, and actionable ACCEPT / REVIEW / REJECT decisions in one pipeline.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Most RAG tutorials stop at: retrieve documents, call the model, read the output.
This library handles what comes next — deciding whether that output should be served,
retried, or regenerated, and telling you exactly why.

Read the full write-up on Towards Data Science →
**[LLM Evals Are Based on Vibes — I Built the Missing Scoring Layer That Actually Decides What to Ship](https://towardsdatascience.com)**

---

## What It Does

```
Query + Context + Response
        ↓
  Scoring Layer
  ├── faithfulness.py   attribution + specificity (hallucination detection built in)
  ├── relevance.py      sentence-level query relevance
  ├── context.py        precision + recall of retrieved context
  └── consistency.py   paraphrase stability (optional, requires llm_fn)
        ↓
  Aggregation Layer
  ├── aggregator.py     weighted score combination + hard floors
  └── llm_judge/        OpenAI fallback for edge cases (0.45–0.65 zone only)
        ↓
  Decision Layer        pipeline.py
  ├── 8 cascaded gates in priority order
  ├── high specificity + low attribution = hallucination
  └── ACCEPT | REVIEW | REJECT
        ↓
  Action Layer
  serve_response | retry_with_specific_prompt | regenerate_with_grounding_prompt
  retrieve_more_documents | optional_human_review
```

| Component | Job |
|---|---|
| `faithfulness.py` | Splits into attribution (grounding) and specificity (concreteness). High specificity + low attribution = hallucination, not weak answer. |
| `relevance.py` | Blends full-response semantic similarity, best sentence match, and token overlap. |
| `context.py` | Precision + recall of retrieved context against the response. Low score → fix retrieval, not the prompt. |
| `consistency.py` | Paraphrase stability across query variants. Requires an LLM callable. Disabled by default. |
| `aggregator.py` | Weighted combination with two hard floors: low relevance penalises final score; low attribution sets a ceiling. |
| `pipeline.py` | Orchestrates all steps. Owns the 8-gate decision logic and confidence calculation. |
| `regression.py` | Stores baselines, diffs current scores, blocks deployment when quality drops beyond threshold. |
| `judge.py` | OpenAI fallback. Only fires in the 0.45–0.65 uncertain zone. Zero API cost in default mode. |

---

## Installation

```bash
git clone https://github.com/Emmimal/llm-eval-layer.git
cd llm-eval-layer
pip install pyyaml                   # required
pip install sentence-transformers    # optional — enables semantic scoring
```

No other dependencies. All core functionality runs on the Python standard library +
PyYAML. If `sentence-transformers` is not installed, the system falls back to TF-IDF
vectors automatically — all functionality remains operational.

For the LLM judge (edge cases only):

```bash
pip install openai
export OPENAI_API_KEY=your_key_here
```

---

## Quick Start

```python
from eval_engine.pipeline import EvalPipeline

pipeline = EvalPipeline(use_llm_judge=False)  # heuristic-only, no API calls

result = pipeline.evaluate(
    query="What is context engineering?",
    context_text="Context engineering is the architectural layer between retrieval "
                 "and generation. It controls what information flows into the LLM "
                 "context window — managing memory, compression, and token budgets.",
    response="Context engineering controls what enters the context window. "
             "It manages memory, compresses context, and enforces token budgets "
             "to keep the model grounded in relevant information.",
)

print(result)
# ────────────────────────────────────────────────────────
#   LLM Eval Result  ✅ PASSED
# ────────────────────────────────────────────────────────
#   Final Score       : 0.680
#   Attribution       : 0.684   (grounding)
#   Specificity       : 0.713   (concreteness)
#   Relevance         : 0.657
#   Context Quality   : 0.688
#   Disagreement      : 0.016   (scorer std dev)
#   ✓  No hallucination
#   Decision          : ✅ ACCEPT  (confidence: 41%)
#   Reason            : All quality gates passed
#   Next Action       : serve_response
#   Latency           : 322ms
# ────────────────────────────────────────────────────────

# JSON-serialisable for logging and dashboards
print(result.to_dict())
```

---

## The Critical Insight: Attribution × Specificity

A single faithfulness score cannot distinguish a hallucination from a weak answer.
This system splits faithfulness into two independent signals:

| | Low Specificity | High Specificity |
|---|---|---|
| **High Attribution** | Grounded but thin → REVIEW | Good answer → ACCEPT |
| **Low Attribution** | Vague, uncertain → REVIEW | **Hallucination → REJECT** |

**High specificity + low attribution = hallucination.** A confident wrong answer is
more dangerous than a vague one — it doesn't signal its own weakness. This is what
a score-only system misses.

```
Attribution : 0.428   (low — poorly grounded in context)
Specificity : 0.701   (high — sounds authoritative and detailed)
Decision    : 🚫 REJECT
Reason      : Confident hallucination detected
Action      : regenerate_with_grounding_prompt
```

---

## Running the Demos

```bash
# Quick start — four examples covering all three decision types
python main.py

# Full RAG evaluation set with benchmark tables
python experiments/rag_eval_demo.py

# Accuracy + latency + regression suite demo
python experiments/benchmarks.py
```

| Script | What It Shows |
|---|---|
| `main.py` | ACCEPT / REVIEW / REJECT across four cases with full output |
| `rag_eval_demo.py` | Full 5-case RAG eval table, decision distribution, before/after comparison |
| `benchmarks.py` | Score separation, hallucination detection rate, latency (10 runs), regression blocking |

---

## Real Benchmark Numbers

**Accuracy (full test set)**

```
Good responses  → mean score : 0.588
Bad responses   → mean score : 0.442
Score separation              : 0.146
Hallucination detection rate  : 2/2  (100%)
```

**Latency (10 runs, warm model, CPU only)**

| Operation | Latency | Notes |
|---|---|---|
| Attribution scorer | ~1.2ms | Embedding + overlap |
| Relevance scorer | ~1.1ms | Sentence-level scoring |
| Context scorer | ~0.8ms | Precision + recall |
| Decision layer | ~0.1ms | Policy rules + confidence |
| Full `pipeline.evaluate()` | ~291ms mean | No LLM calls |
| With LLM judge | ~340ms | Edge cases only (0.45–0.65 zone) |

First run is slower due to `sentence-transformers` model loading. Subsequent calls
average 291ms. In production with the model pre-loaded at startup, the full
evaluation layer adds under 300ms per response.

**RAG evaluation across the full test set**

```
+-------+-----------------------+-------+-------+-------+-------+---------------+-----------+
| ID    | Label                 | Attr  | Relev | Ctx   | Final | Hallucination | Decision  |
+-------+-----------------------+-------+-------+-------+-------+---------------+-----------+
| q_001 | good_response         | 0.686 | 0.680 | 0.725 | 0.694 | ✓  No         | ✅ ACCEPT  |
| q_002 | hallucinated_response | 0.445 | 0.621 | 0.459 | 0.547 | ⚠️  Suspected | 🚫 REJECT  |
| q_003 | good_response         | 0.528 | 0.456 | 0.535 | 0.534 | ⚠️  Suspected | 🔍 REVIEW  |
| q_004 | off_context_response  | 0.043 | 0.682 | 0.091 | 0.337 | 🚫 Confirmed  | 🚫 REJECT  |
| q_005 | good_response         | 0.625 | 0.341 | 0.628 | 0.536 | ✓  No         | 🔍 REVIEW  |
+-------+-----------------------+-------+-------+-------+-------+---------------+-----------+
```

---

## The Regression Suite

The most valuable component. Store baselines after validating your system, then run
regression checks after every prompt change, model update, or retrieval change. If
any case drops more than 5 points, deployment blocks.

```python
from eval_engine.regression import RegressionSuite

suite = RegressionSuite(store_path="data/baselines.json", regression_threshold=0.05)

# After validating your system
suite.record_baseline("q_001", query, context, response, result)

# After changing your prompt or model
report = suite.run_regression(pipeline, updated_test_cases)

if report.failed > 0:
    raise SystemExit("Quality regression detected. Deployment blocked.")
```

```
════════════════════════════════════════════════════
  Regression Report  —  CI/CD Quality Gate
════════════════════════════════════════════════════
  🚫 3 REGRESSION(S) DETECTED — DEPLOYMENT BLOCKED
────────────────────────────────────────────────────
  Total cases   : 3   |   Passed : 0   |   Failed : 3
  Mean Δ score  : -0.4586
  Threshold     : ±0.05

  [q_001] 0.694 → 0.137 (Δ -0.556)
  [q_002] 0.547 → 0.137 (Δ -0.409)
  [q_003] 0.534 → 0.124 (Δ -0.410)
════════════════════════════════════════════════════
```

This is CI/CD for LLMs. Not "check if it looks right" — fail the build when quality
drops beyond a threshold, exactly like a failing unit test.

---

## Configuration

**`configs/weights.yaml`** — scoring weights (must sum to 1.0)

```yaml
faithfulness:    0.40   # increase for RAG systems
relevance:       0.30   # increase for chatbots
context_quality: 0.20
consistency:     0.10
```

> Note: the aggregator code uses updated defaults of 0.45 / 0.25 for faithfulness
> and relevance, which produced better score separation on the evaluation set.
> The YAML values are the starting point — the in-code defaults take precedence
> until you override them in the file.

**`configs/thresholds.yaml`** — decision gates

```yaml
reject_threshold:          0.45   # below this → REJECT
review_threshold:          0.65   # below this → REVIEW; above → ACCEPT
faithfulness_min:          0.40
relevance_min:             0.45
context_quality_min:       0.35
llm_escalation_threshold:  0.50   # LLM judge fires between 0.45–0.65
final_score_min:           0.45
```

No code changes needed to tune either file. Tighten thresholds for higher-stakes
domains (medical, legal). Loosen them for domains that tolerate more ambiguity.

---

## EvalPipeline API

```python
EvalPipeline(
    use_llm_judge=False,   # enable LLM fallback for 0.45–0.65 zone
    llm_fn=None,           # callable(query) → response, for consistency scoring
    api_key=None,          # OpenAI key — falls back to OPENAI_API_KEY env var
)

pipeline.evaluate(
    query="...",
    context_text="...",
    response="...",
    run_consistency=False,  # enable consistency scorer (requires llm_fn)
)
```

**`EvalResult` fields**

| Field | Type | Description |
|---|---|---|
| `decision` | str | `ACCEPT` / `REVIEW` / `REJECT` |
| `confidence_pct` | int | 0–100, blends margin + attribution + scorer stability |
| `failure_type` | str | `none` / `hallucination` / `weak_grounding` / `poor_retrieval` / `off_topic` / `uncertain` |
| `hallucination_status` | str | `none` / `suspected` / `confirmed` |
| `action` | str | Next action: `serve_response`, `retry_with_specific_prompt`, etc. |
| `attribution_score` | float | Grounding in context (0–1) |
| `specificity_score` | float | Concreteness of response (0–1) |
| `relevance_score` | float | Query relevance (0–1) |
| `context_quality_score` | float | Retrieval quality (0–1) |
| `disagreement` | float | Scorer std dev — high = system uncertain |
| `final_score` | float | Weighted composite (0–1) |
| `low_confidence_sentences` | list | Sentences below attribution threshold |
| `latency_ms` | float | Full pipeline wall time |

```python
# JSON-serialisable for logging, APIs, and dashboards
result.to_dict()
# {
#   "decision": "REJECT",
#   "confidence_pct": 22,
#   "failure_type": "hallucination",
#   "hallucination_status": "suspected",
#   "next_action": "regenerate_with_grounding_prompt",
#   "scores": {
#     "final": 0.525, "attribution": 0.428, "specificity": 0.701,
#     "relevance": 0.613, "context_quality": 0.424, "disagreement": 0.077
#   },
#   "explanations": {
#     "reason": "Confident hallucination detected...",
#     "low_confidence_sentences": ["It has nothing to do with language models."]
#   },
#   "meta": { "passed": false, "used_llm_judge": false, "latency_ms": 301.0 }
# }
```

---

## Unit Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_eval.py
```

The test suite covers all four scorers, the aggregator, and the full pipeline —
including hallucination detection, score separation, disagreement calculation,
and `to_dict()` serialisation.

---

## Project Structure

```
llm-eval-layer/
├── eval_engine/
│   ├── pipeline.py       # EvalPipeline — orchestrates all steps + decision layer
│   ├── aggregator.py     # weighted score combination + hard floors
│   └── regression.py    # CI/CD regression suite
├── scorers/
│   ├── faithfulness.py   # attribution + specificity (hallucination detection)
│   ├── relevance.py      # sentence-level query relevance
│   ├── context.py        # context precision + recall
│   └── consistency.py    # paraphrase stability (optional)
├── heuristics/
│   ├── embeddings.py     # sentence-transformers with TF-IDF fallback
│   └── similarity.py     # cosine, token overlap, recall, precision helpers
├── llm_judge/
│   ├── judge.py          # OpenAI fallback (edge cases only)
│   └── prompts.py        # versioned prompt templates
├── configs/
│   ├── weights.yaml      # tunable scoring weights
│   └── thresholds.yaml   # decision gate thresholds
├── data/
│   └── sample_queries.json
├── experiments/
│   ├── rag_eval_demo.py  # full RAG benchmark tables
│   └── benchmarks.py     # latency + accuracy + regression demo
├── tests/
│   └── test_eval.py
└── main.py               # quick start
```

---

## When to Use This

Worth building when you have:

- RAG systems where hallucination is a real consequence, not just a metric
- Automated pipelines with no human review step between generation and delivery
- Prompt engineering workflows where you need to know if a change made things worse
- Any system where "the output looked fine" is not a sufficient QA process

Skip it when you have:

- Single-turn demos with no production traffic
- Human review on every response regardless
- Fully deterministic domains where exact-match evaluation is sufficient
- Hard latency requirements under 50ms (use TF-IDF mode for ~5ms total)

---

## Known Limitations

**Subtle factual errors.** Attribution is a grounding signal, not a fact-checker.
A response that mirrors context vocabulary but substitutes one wrong number can score
well on attribution. The LLM judge catches this in edge cases — heuristics alone
do not.

**Token estimation.** Uses 1 token ≈ 4 characters. Misfires for code-heavy
responses and non-Latin scripts. Swap in `tiktoken` in `heuristics/embeddings.py`
for exact counts.

**Extractive consistency.** The consistency scorer paraphrases queries with simple
rule-based rewrites, not semantic paraphrases. Variants like "Can you explain X?"
and "Tell me about X" are lightweight approximations. For production consistency
testing, provide an `llm_fn` that calls your actual model.

**Threshold calibration.** `configs/thresholds.yaml` is calibrated on a small
general-purpose evaluation set. Run `experiments/benchmarks.py` against your own
labeled cases and tune before trusting any value listed here. A medical QA system
needs tighter hallucination thresholds than a creative writing assistant.

**`embeddings.position_ids` warning on load.** Cosmetic. A known artifact of
loading `all-MiniLM-L6-v2` from a sentence-transformers checkpoint. No impact on
scoring accuracy or runtime performance.

---

## License

MIT
