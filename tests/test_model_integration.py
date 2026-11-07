"""Integration test with the REAL embedding model (needs sentence-transformers + internet on first run).

Run:  python -m pytest tests/test_model_integration.py -q -s
Skipped automatically if sentence-transformers is not installed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("sentence_transformers")

from sentence_transformers import SentenceTransformer  # noqa: E402

from utils.data import DEMO_DOCS, DEMO_QUERY  # noqa: E402
from utils.search import doc_text, semantic_search  # noqa: E402
from utils.similarity import cosine_similarity  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_shape(model):
    vecs = model.encode(["hello world", "another sentence"])
    assert vecs.shape == (2, 384)


def test_similar_sentences_score_higher_than_unrelated(model):
    a, b, c = model.encode([
        "I'm exhausted because I stayed up all night studying.",
        "I feel very tired after an all-night study session.",
        "Photosynthesis converts sunlight into chemical energy.",
    ])
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


def test_module1_semantic_beats_keyword(model):
    """The lesson of Module 1: meaning-based search ranks a relevant doc above the battery guide."""
    doc_vecs = model.encode([doc_text(d) for d in DEMO_DOCS])
    query_vec = model.encode([DEMO_QUERY])[0]
    ranked = semantic_search(query_vec, doc_vecs, DEMO_DOCS, top_k=4)
    titles = [r["doc"]["title"] for r in ranked]
    assert titles.index("How to speed up a slow PC") < titles.index("Laptop battery maintenance guide")
    assert np.all(np.diff([r["score"] for r in ranked]) <= 1e-9)  # sorted best-first
