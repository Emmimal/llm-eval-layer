"""
Answer Relevance Scorer
=======================
Measures how directly the LLM response answers the original query.

A relevant response addresses what was asked — not adjacent topics,
not over-generalizations, not restatements of the question.

Score range: 0.0 (completely off-topic) → 1.0 (directly answers the query)
"""

from __future__ import annotations
from dataclasses import dataclass

from heuristics.similarity import (
    semantic_similarity,
    max_sentence_similarity,
    token_overlap,
)


@dataclass
class RelevanceResult:
    score: float           # composite relevance score
    semantic_score: float  # full-response semantic similarity to query
    max_sent_score: float  # best single-sentence match to query
    overlap_score: float   # lexical overlap between query and response
    is_relevant: bool      # True when score >= threshold


_RELEVANCE_THRESHOLD = 0.45


def score(query: str, response: str) -> RelevanceResult:
    """
    Score how relevant *response* is to *query*.

    Composite blends:
      - full-response semantic similarity (weight 0.45)
      - best sentence-level match        (weight 0.35)
      - token overlap                    (weight 0.20)

    The sentence-level component rewards focused answers: a response
    that contains at least one sentence that directly answers the
    question scores well even if the overall response is verbose.
    """
    if not query.strip() or not response.strip():
        return RelevanceResult(
            score=0.0,
            semantic_score=0.0,
            max_sent_score=0.0,
            overlap_score=0.0,
            is_relevant=False,
        )

    semantic  = semantic_similarity(query, response)
    max_sent  = max_sentence_similarity(query, response)
    overlap   = token_overlap(query, response)

    composite = 0.45 * semantic + 0.35 * max_sent + 0.20 * overlap

    return RelevanceResult(
        score=round(composite, 4),
        semantic_score=round(semantic, 4),
        max_sent_score=round(max_sent, 4),
        overlap_score=round(overlap, 4),
        is_relevant=composite >= _RELEVANCE_THRESHOLD,
    )
