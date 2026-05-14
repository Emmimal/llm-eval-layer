"""
Faithfulness Scorer
===================
Measures how well the LLM response is grounded in the provided context.

Split into TWO distinct signals:

  Attribution Score  — "Is the answer supported by the context?"
                       Semantic + lexical overlap between context and response.
                       Low attribution = hallucination risk.

  Specificity Score  — "Is the answer concrete or vague?"
                       Measures response density: length, distinct terms,
                       absence of hedge phrases. Low specificity = weak answer.

Why split?
  A hallucination is not the same as a weak answer.
  Attribution catches fabrication. Specificity catches vagueness.
  Conflating them causes false positives and false negatives.

Score range: 0.0 → 1.0 for each signal and composite.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from heuristics.similarity import (
    sentence_split,
    semantic_similarity,
    token_overlap,
)


@dataclass
class FaithfulnessResult:
    score: float                        # composite (attribution + specificity)
    attribution_score: float            # is the answer supported by context?
    specificity_score: float            # is the answer concrete vs vague?
    semantic_score: float               # embedding-based grounding
    overlap_score: float                # lexical grounding
    sentence_scores: List[float]        # per-sentence attribution score
    hallucination_detected: bool        # True when attribution < threshold
    low_confidence_sentences: List[str] # sentences likely to be hallucinated


# ── Thresholds ────────────────────────────────────────────────────────────────
_HALLUCINATION_THRESHOLD = 0.55   # attribution below this → flag hallucination
_SENTENCE_THRESHOLD      = 0.35   # per-sentence attribution threshold

# Hedge phrases that signal vagueness / low specificity
_HEDGE_PHRASES = [
    "it can be", "it may be", "in various", "many scenarios",
    "could be used", "might be", "generally speaking", "in some cases",
    "it depends", "various ways", "can help with",
]


def _specificity_score(response: str) -> float:
    """
    Score how specific/concrete the response is.

    Three factors:
      1. Length density  — longer responses tend to be more specific
      2. Vocabulary richness — distinct tokens / total tokens
      3. Hedge penalty   — deduct for vague filler phrases

    Returns 0.0 (maximally vague) → 1.0 (maximally specific)
    """
    if not response.strip():
        return 0.0

    tokens = re.findall(r"[a-z]+", response.lower())
    if not tokens:
        return 0.0

    # 1. Length score: saturates at ~80 tokens (typical good answer length)
    length_score = min(1.0, len(tokens) / 80)

    # 2. Vocabulary richness: type-token ratio
    richness = len(set(tokens)) / len(tokens)

    # 3. Hedge penalty: -0.15 per hedge phrase found
    hedge_count = sum(1 for h in _HEDGE_PHRASES if h in response.lower())
    hedge_penalty = min(0.6, hedge_count * 0.15)

    raw = (0.40 * length_score + 0.60 * richness) - hedge_penalty
    return round(max(0.0, min(1.0, raw)), 4)


def score(context: str, response: str) -> FaithfulnessResult:
    """
    Score faithfulness as attribution + specificity.

    Composite:
      0.70 * attribution + 0.30 * specificity

    Attribution dominates because grounding is the primary signal.
    Specificity breaks ties between similarly-grounded responses.
    """
    if not context.strip() or not response.strip():
        return FaithfulnessResult(
            score=0.0,
            attribution_score=0.0,
            specificity_score=0.0,
            semantic_score=0.0,
            overlap_score=0.0,
            sentence_scores=[],
            hallucination_detected=True,
            low_confidence_sentences=[],
        )

    # ── Attribution ──────────────────────────────────────────────────────────
    semantic    = semantic_similarity(context, response)
    overlap     = token_overlap(context, response)
    attribution = round(0.60 * semantic + 0.40 * overlap, 4)

    # ── Specificity ──────────────────────────────────────────────────────────
    specificity = _specificity_score(response)

    # ── Composite ────────────────────────────────────────────────────────────
    composite = round(0.70 * attribution + 0.30 * specificity, 4)

    # ── Per-sentence attribution ──────────────────────────────────────────────
    sentences = sentence_split(response)
    sentence_scores = [semantic_similarity(context, s) for s in sentences]
    low_confidence  = [
        s for s, sc in zip(sentences, sentence_scores)
        if sc < _SENTENCE_THRESHOLD
    ]

    return FaithfulnessResult(
        score=composite,
        attribution_score=attribution,
        specificity_score=specificity,
        semantic_score=round(semantic, 4),
        overlap_score=round(overlap, 4),
        sentence_scores=[round(s, 4) for s in sentence_scores],
        hallucination_detected=attribution < _HALLUCINATION_THRESHOLD,
        low_confidence_sentences=low_confidence,
    )
