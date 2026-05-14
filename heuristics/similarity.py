"""
Lexical and semantic similarity helpers used across all scorers.
"""

from __future__ import annotations
import re
from typing import List, Set

from heuristics.embeddings import get_embedding, cosine_similarity


def token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    ta: Set[str] = set(re.findall(r"[a-z]+", a.lower()))
    tb: Set[str] = set(re.findall(r"[a-z]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sentence_split(text: str) -> List[str]:
    """Split text into sentences on . ! ? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of embeddings for two strings."""
    return cosine_similarity(get_embedding(a), get_embedding(b))


def max_sentence_similarity(query: str, text: str) -> float:
    """
    Return the maximum semantic similarity between *query* and any
    individual sentence in *text*.  Useful for relevance scoring when
    the answer is a multi-sentence paragraph.
    """
    sentences = sentence_split(text)
    if not sentences:
        return 0.0
    return max(semantic_similarity(query, s) for s in sentences)


def recall(reference: str, candidate: str) -> float:
    """
    Token recall: fraction of reference tokens present in candidate.
    Used for faithfulness (how much of the context is covered).
    """
    ref_tokens = set(re.findall(r"[a-z]+", reference.lower()))
    cand_tokens = set(re.findall(r"[a-z]+", candidate.lower()))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def precision(reference: str, candidate: str) -> float:
    """
    Token precision: fraction of candidate tokens that appear in reference.
    Used for context precision scoring.
    """
    ref_tokens = set(re.findall(r"[a-z]+", reference.lower()))
    cand_tokens = set(re.findall(r"[a-z]+", candidate.lower()))
    if not cand_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(cand_tokens)
