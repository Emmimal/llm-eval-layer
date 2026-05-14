"""
Context Quality Scorer
======================
Measures how well the retrieved context serves the query and response.

Two dimensions:
  - Context Precision : fraction of context tokens that are useful
                        (i.e. appear in the response)
  - Context Recall    : fraction of response tokens that are grounded
                        in the context

Together they reveal retrieval noise (low precision) and coverage
gaps (low recall).

Score range: 0.0 → 1.0 for each dimension and composite F1.
"""

from __future__ import annotations
from dataclasses import dataclass

from heuristics.similarity import precision, recall, semantic_similarity


@dataclass
class ContextResult:
    precision_score: float   # how much of the context is actually used
    recall_score: float      # how much of the response is grounded
    f1_score: float          # harmonic mean (composite context quality)
    semantic_score: float    # semantic alignment between context and response
    score: float             # final blended score


def score(context: str, response: str) -> ContextResult:
    """
    Score context quality given *context* and LLM *response*.

    Precision and recall use token overlap; the composite blends
    the F1 of those two with embedding-based semantic alignment.

    Composite:
      0.50 * F1 + 0.50 * semantic_similarity
    """
    if not context.strip() or not response.strip():
        return ContextResult(
            precision_score=0.0,
            recall_score=0.0,
            f1_score=0.0,
            semantic_score=0.0,
            score=0.0,
        )

    prec = precision(context, response)   # context → response coverage
    rec  = recall(response, context)      # response → context grounding

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    semantic = semantic_similarity(context, response)
    composite = 0.50 * f1 + 0.50 * semantic

    return ContextResult(
        precision_score=round(prec, 4),
        recall_score=round(rec, 4),
        f1_score=round(f1, 4),
        semantic_score=round(semantic, 4),
        score=round(composite, 4),
    )
