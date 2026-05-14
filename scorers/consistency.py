"""
Consistency Scorer
==================
Measures whether the LLM produces stable answers across paraphrased
versions of the same query.

An inconsistent model is an unreliable model — even if each individual
response scores well on faithfulness and relevance.

How it works:
  1. Generate N paraphrases of the original query (rule-based, no LLM).
  2. Score each paraphrase response against the original response.
  3. Consistency = mean pairwise semantic similarity across all responses.

Score range: 0.0 (completely inconsistent) → 1.0 (perfectly stable)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable

from heuristics.similarity import semantic_similarity


@dataclass
class ConsistencyResult:
    score: float                        # mean pairwise similarity
    pairwise_scores: List[float]        # similarity of each variant to original
    num_variants: int
    is_consistent: bool                 # True when score >= threshold


_CONSISTENCY_THRESHOLD = 0.70

# ── Simple rule-based paraphrasers (no LLM required) ────────────────────────

def _paraphrase_variants(query: str) -> List[str]:
    """
    Generate lightweight paraphrases by surface-level rewrites.
    These are not perfect — but they're fast, free, and dependency-free.
    """
    q = query.strip().rstrip("?")
    variants = [
        f"Can you explain {q.lower()}?",
        f"Tell me about {q.lower()}.",
        f"What do you know about {q.lower()}?",
    ]
    return variants


def score(
    query: str,
    original_response: str,
    llm_fn: Callable[[str], str],
    num_variants: int = 3,
) -> ConsistencyResult:
    """
    Score response consistency.

    Parameters
    ----------
    query             : the original query string
    original_response : the LLM's response to the original query
    llm_fn            : callable(query) → response string
                        (plug in your LLM call here)
    num_variants      : how many paraphrase variants to test (max 3)
    """
    variants = _paraphrase_variants(query)[:num_variants]

    pairwise: List[float] = []
    for v in variants:
        variant_response = llm_fn(v)
        sim = semantic_similarity(original_response, variant_response)
        pairwise.append(round(sim, 4))

    avg = sum(pairwise) / len(pairwise) if pairwise else 0.0

    return ConsistencyResult(
        score=round(avg, 4),
        pairwise_scores=pairwise,
        num_variants=len(variants),
        is_consistent=avg >= _CONSISTENCY_THRESHOLD,
    )
