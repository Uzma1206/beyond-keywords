"""Unit tests for the pure logic (no Streamlit, no model download needed).

Run:  python -m pytest tests/test_logic.py -q      (or: python tests/test_logic.py)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import DEMO_DOCS, DEMO_QUERY, PLAYGROUND_DOCS, UNIVERSITY_DOCS  # noqa: E402
from utils.search import (  # noqa: E402
    build_rag_prompt, extract_keywords, is_valid_document, keyword_search, prepare_documents,
    semantic_search, shared_terms,
)
from utils.similarity import (  # noqa: E402
    cosine_similarity, cosine_similarity_matrix, interpret_similarity, pairwise_cosine_matrix, rank_by_score,
)


# ------------------------------------------------------------------ cosine similarity
def test_identical_vectors_score_one():
    assert math.isclose(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)


def test_opposite_vectors_score_minus_one():
    assert math.isclose(cosine_similarity([1, 2], [-1, -2]), -1.0)


def test_orthogonal_vectors_score_zero():
    assert math.isclose(cosine_similarity([1, 0], [0, 5]), 0.0, abs_tol=1e-12)


def test_length_does_not_change_score():
    assert math.isclose(cosine_similarity([1, 2], [3, 6]), 1.0)
    assert math.isclose(cosine_similarity([1, 2], [2, 3]), 8 / (math.sqrt(5) * math.sqrt(13)))


def test_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert list(cosine_similarity_matrix([0, 0], [[1, 2], [3, 4]])) == [0.0, 0.0]


def test_dimension_mismatch_raises():
    for bad in (lambda: cosine_similarity([1, 2], [1, 2, 3]),
                lambda: cosine_similarity_matrix([1, 2], [[1, 2, 3]])):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_matrix_version_matches_scalar_version():
    rng = np.random.default_rng(0)
    q, docs = rng.normal(size=384), rng.normal(size=(7, 384))
    fast = cosine_similarity_matrix(q, docs)
    slow = [cosine_similarity(q, d) for d in docs]
    assert np.allclose(fast, slow)


def test_pairwise_matrix_is_symmetric_with_unit_diagonal():
    m = pairwise_cosine_matrix(np.random.default_rng(1).normal(size=(5, 16)))
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)


def test_empty_collection():
    assert len(cosine_similarity_matrix([1, 2], np.zeros((0, 2)))) == 0
    assert semantic_search([1, 2], np.zeros((0, 2)), [], top_k=5) == []


# ------------------------------------------------------------------ ranking
def test_ranking_orders_best_first():
    assert rank_by_score([0.1, 0.9, 0.5, 0.7], top_k=3) == [1, 3, 2]


def test_ranking_ties_keep_original_order():
    assert rank_by_score([0.5, 0.5, 0.5], top_k=2) == [0, 1]


def test_semantic_search_ranks_by_cosine_and_respects_top_k():
    docs = [{"title": t, "category": "x", "text": "some text here"} for t in "ABCD"]
    vecs = np.array([[0, 1], [1, 0], [1, 1], [-1, 0]], dtype=float)
    res = semantic_search([1, 0.1], vecs, docs, top_k=2)
    assert [r["doc"]["title"] for r in res] == ["B", "C"]
    assert res[0]["score"] > res[1]["score"]
    assert semantic_search([1, 0.1], vecs, docs, top_k=4, min_score=0.9)[0]["doc"]["title"] == "B"


def test_semantic_search_length_mismatch_raises():
    try:
        semantic_search([1, 0], np.zeros((2, 2)), [{"title": "a", "text": "abcdefghij"}], top_k=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_interpretation_bands():
    assert interpret_similarity(0.8)[0] == "Very similar"
    assert interpret_similarity(0.5)[0] == "Related"
    assert interpret_similarity(0.3)[0] == "Loosely related"
    assert interpret_similarity(0.0)[0] == "Mostly unrelated"


# ------------------------------------------------------------------ keyword search
def test_keywords_drop_stopwords_and_dedupe():
    assert extract_keywords(DEMO_QUERY) == ["make", "laptop", "faster"]
    assert extract_keywords("the the The of") == []


def test_module1_story_keyword_search_fails_as_taught():
    res = keyword_search(DEMO_QUERY, DEMO_DOCS)
    assert [r["doc"]["title"] for r in res] == ["Laptop battery maintenance guide"]
    titles = {d["title"] for d in DEMO_DOCS}
    assert "How to speed up a slow PC" in titles  # exists, but keyword search misses it


def test_keyword_score_is_fraction_of_query_terms():
    docs = [{"title": "Money", "category": "f", "text": "save money every month for school"}]
    res = keyword_search("save money fast", docs)
    assert math.isclose(res[0]["score"], 2 / 3) and res[0]["matched_terms"] == ["save", "money"]


def test_keyword_is_exact_not_stemmed():
    docs = [{"title": "Speed", "category": "t", "text": "make it fast and quick right now"}]
    assert keyword_search("faster", docs) == []


def test_keyword_empty_and_stopword_only_queries():
    assert keyword_search("", PLAYGROUND_DOCS) == []
    assert keyword_search("how to the", PLAYGROUND_DOCS) == []


def test_shared_terms():
    assert shared_terms("hackers online", PLAYGROUND_DOCS[5]) == ["hackers", "online"]


# ------------------------------------------------------------------ documents & data
def test_document_validation():
    assert is_valid_document({"title": "t", "text": "A real sentence of text."})
    for bad in ({"text": ""}, {"text": "   "}, {"text": "short"}, {"text": "12345 67890 !!!"}, None, "string"):
        assert not is_valid_document(bad)


def test_prepare_documents_skips_invalid():
    docs, skipped = prepare_documents([{"text": "valid document text"}, {"text": ""}, {"text": "??"}])
    assert len(docs) == 1 and skipped == 2


def test_sample_collections_meet_brief():
    assert len(PLAYGROUND_DOCS) >= 10
    assert len({d["category"] for d in PLAYGROUND_DOCS}) >= 6
    assert len(UNIVERSITY_DOCS) >= 5
    assert all(is_valid_document(d) for d in PLAYGROUND_DOCS + UNIVERSITY_DOCS + DEMO_DOCS)


def test_rag_prompt_contains_context_and_question():
    res = [{"doc": UNIVERSITY_DOCS[0], "score": 0.5}]
    prompt = build_rag_prompt("Is attendance required?", res)
    assert UNIVERSITY_DOCS[0]["title"] in prompt and "Is attendance required?" in prompt


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} tests passed")
