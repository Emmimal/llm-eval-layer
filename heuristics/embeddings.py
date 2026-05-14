"""
Embedding generation for the LLM Eval Layer.
Uses sentence-transformers if available; falls back to TF-IDF vectors.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_MODE = True
except ImportError:
    _MODEL = None
    SEMANTIC_MODE = False


def get_embedding(text: str) -> List[float]:
    """Return a vector for *text*. Uses sentence-transformers when available."""
    if SEMANTIC_MODE and _MODEL is not None:
        return _MODEL.encode(text, normalize_embeddings=True).tolist()
    return _tfidf_vector(text)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ── TF-IDF fallback ──────────────────────────────────────────────────────────

_IDF_CACHE: dict[str, float] = {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_vector(text: str) -> List[float]:
    tokens = _tokenize(text)
    tf = Counter(tokens)
    vocab = sorted(tf.keys())
    vec = []
    for term in vocab:
        idf = _IDF_CACHE.get(term, math.log(10))          # default IDF = log(10)
        vec.append((tf[term] / max(len(tokens), 1)) * idf)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def update_idf(corpus: List[str]) -> None:
    """Pre-compute IDF from a document corpus for better TF-IDF accuracy."""
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for term in set(_tokenize(doc)):
            df[term] += 1
    _IDF_CACHE.clear()
    for term, freq in df.items():
        _IDF_CACHE[term] = math.log((N + 1) / (freq + 1)) + 1
