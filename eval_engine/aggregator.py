"""
Score Aggregator
================
Combines individual dimension scores into a single final quality score.

Design:
  - Weights are loaded from configs/weights.yaml (tunable per use case)
  - Consistency is optional (requires an LLM callable to compute)
  - The aggregator also decides whether LLM escalation is needed
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from scorers.faithfulness import FaithfulnessResult
from scorers.relevance import RelevanceResult
from scorers.context import ContextResult


# ── Default weights ──────────────────────────────────────────────────────────
# Attribution (faithfulness) is the strongest signal — it dominates.
# Increasing its weight widens separation between good and bad responses.
_DEFAULT_WEIGHTS = {
    "faithfulness":    0.45,   # attribution-heavy (was 0.40)
    "relevance":       0.25,   # slightly reduced  (was 0.30)
    "context_quality": 0.20,
    "consistency":     0.10,
}

_DEFAULT_THRESHOLDS = {
    "llm_escalation_threshold": 0.50,
    "final_score_min":          0.50,
}


def _load_yaml(path: str) -> dict:
    if not _YAML_AVAILABLE:
        return {}
    full = os.path.join(os.path.dirname(__file__), "..", path)
    try:
        with open(full) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


@dataclass
class AggregatedScore:
    final_score: float
    faithfulness: float
    relevance: float
    context_quality: float
    consistency: Optional[float]
    hallucination_detected: bool
    needs_llm_review: bool
    passed: bool
    low_confidence_sentences: list = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)


def aggregate(
    faithfulness_result: FaithfulnessResult,
    relevance_result: RelevanceResult,
    context_result: ContextResult,
    consistency_score: Optional[float] = None,
) -> AggregatedScore:
    """
    Combine dimension scores into a final quality score.

    If consistency_score is None (not computed), its weight is
    redistributed proportionally to the other three dimensions.
    """
    weights = {**_DEFAULT_WEIGHTS, **_load_yaml("configs/weights.yaml")}
    thresholds = {**_DEFAULT_THRESHOLDS, **_load_yaml("configs/thresholds.yaml")}

    f  = faithfulness_result.score
    r  = relevance_result.score
    c  = context_result.score
    co = consistency_score

    if co is None:
        base = weights["faithfulness"] + weights["relevance"] + weights["context_quality"]
        wf = weights["faithfulness"] / base
        wr = weights["relevance"] / base
        wc = weights["context_quality"] / base
        final = wf * f + wr * r + wc * c
    else:
        final = (
            weights["faithfulness"]    * f +
            weights["relevance"]       * r +
            weights["context_quality"] * c +
            weights["consistency"]     * co
        )

    # ── Hard floor: low relevance penalises final score ───────────────────────
    # A response that doesn't answer the question is bad regardless of grounding.
    if r < 0.30:
        final *= 0.70

    # ── Hard floor: attribution floor sets a score ceiling ────────────────────
    # A response with very low attribution cannot score well overall.
    if f < 0.35:
        final = min(final, 0.40)

    needs_review = (
        f < thresholds["llm_escalation_threshold"] or
        r < thresholds["llm_escalation_threshold"]
    )

    return AggregatedScore(
        final_score=round(final, 4),
        faithfulness=round(f, 4),
        relevance=round(r, 4),
        context_quality=round(c, 4),
        consistency=round(co, 4) if co is not None else None,
        hallucination_detected=faithfulness_result.hallucination_detected,
        needs_llm_review=needs_review,
        passed=final >= thresholds["final_score_min"],
        low_confidence_sentences=faithfulness_result.low_confidence_sentences,
        weights_used=weights,
    )
