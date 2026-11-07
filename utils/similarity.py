"""Pure NumPy similarity maths.

This file deliberately does NOT import Streamlit or the embedding model, so the
core maths can be unit-tested on its own (see tests/test_logic.py).

    similarity(A, B) = (A . B) / (||A|| x ||B||)
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 if either vector has zero length (the angle is undefined),
    instead of crashing with a division-by-zero error.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have the same length, got {a.shape[0]} and {b.shape[0]}.")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def cosine_similarity_matrix(query_vec, doc_matrix) -> np.ndarray:
    """Cosine similarity of ONE query vector against MANY document vectors.

    doc_matrix has shape (n_documents, n_dimensions). Returns an array of
    n_documents scores, in the same order as the documents.
    """
    q = np.asarray(query_vec, dtype=np.float64).ravel()
    m = np.asarray(doc_matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if m.shape[1] != q.shape[0]:
        raise ValueError(
            f"Query has {q.shape[0]} dimensions but documents have {m.shape[1]}. "
            "Queries and documents must be embedded with the same model."
        )
    dots = m @ q                                      # A . B for every document
    denominators = np.linalg.norm(m, axis=1) * np.linalg.norm(q)   # ||A|| x ||B||
    scores = np.zeros_like(dots)
    valid = denominators > 0
    scores[valid] = dots[valid] / denominators[valid]
    return np.clip(scores, -1.0, 1.0)


def pairwise_cosine_matrix(vectors) -> np.ndarray:
    """Cosine similarity between every pair of vectors (n x n matrix)."""
    m = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = m / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def rank_by_score(scores, top_k: int = 5) -> list[int]:
    """Indices of the top_k highest scores, best first. Ties keep original order."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return [int(i) for i in order[: max(0, int(top_k))]]


def interpret_similarity(score: float) -> tuple[str, str]:
    """Plain-language label for a cosine score.

    These bands are rough rules of thumb for all-MiniLM-L6-v2 on short English
    text. They are NOT universal thresholds: other models use different scales.
    """
    if score >= 0.65:
        return "Very similar", "The two texts are about almost the same thing."
    if score >= 0.45:
        return "Related", "The texts share a clear topic or idea."
    if score >= 0.25:
        return "Loosely related", "There is some overlap in topic, but not much."
    return "Mostly unrelated", "The model sees little shared meaning."
