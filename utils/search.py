"""Keyword search, semantic search and small retrieval helpers.

Documents are plain dicts: {"title": str, "category": str, "text": str}.
The functions here do not import Streamlit at module level, so they can be
tested without it. The one function that needs the embedding model imports it
lazily.
"""
from __future__ import annotations

import re

from utils.similarity import cosine_similarity_matrix, rank_by_score

# Common words that carry little meaning. Keyword engines usually ignore them.
STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from up down out over under
    about into as is are was were be been being am do does did doing have has had
    having i me my mine we our you your he she it its they them their this that
    these those what which who whom how why when where can could should would will
    may might must not no so than too very just also there here s t""".split()
)

MIN_DOC_CHARS = 10


# --------------------------------------------------------------------------- #
# Document helpers
# --------------------------------------------------------------------------- #
def doc_text(doc: dict) -> str:
    """The text we search over / embed: title + body."""
    title = str(doc.get("title", "")).strip()
    body = str(doc.get("text", "")).strip()
    return f"{title}. {body}" if title else body


def is_valid_document(doc) -> bool:
    """A document is valid if it has some real text (letters, not just symbols)."""
    if not isinstance(doc, dict):
        return False
    body = str(doc.get("text", "") or "").strip()
    return len(body) >= MIN_DOC_CHARS and re.search(r"[A-Za-z]", body) is not None


def prepare_documents(documents) -> tuple[list[dict], int]:
    """Drop invalid documents. Returns (valid_documents, number_skipped)."""
    valid = [d for d in documents if is_valid_document(d)]
    return valid, len(documents) - len(valid)


# --------------------------------------------------------------------------- #
# Keyword (lexical) search
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Lower-case a text and split it into word tokens (no stemming)."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text or "").lower())


def extract_keywords(text: str) -> list[str]:
    """Unique, order-preserving content words (stopwords removed)."""
    seen: list[str] = []
    for token in tokenize(text):
        if token not in STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def keyword_search(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """Rank documents by the share of query keywords that appear in them.

    score = (query keywords found in the document) / (query keywords)

    This is a KEYWORD-MATCH score between 0 and 1. It is NOT a cosine
    similarity and says nothing about meaning. Words must match exactly:
    "faster" does not match "fast", and "laptop" does not match "computer".
    Documents with no matching word are not returned at all.
    """
    query_terms = extract_keywords(query)
    if not query_terms:
        return []
    results = []
    for index, doc in enumerate(documents):
        doc_terms = set(tokenize(doc_text(doc)))
        matched = [t for t in query_terms if t in doc_terms]
        if matched:
            results.append(
                {
                    "index": index,
                    "doc": doc,
                    "score": len(matched) / len(query_terms),
                    "matched_terms": matched,
                }
            )
    results.sort(key=lambda r: (-r["score"], r["index"]))
    return results[: max(0, int(top_k))]


def shared_terms(query: str, doc: dict) -> list[str]:
    """Query keywords that literally appear in a document."""
    doc_terms = set(tokenize(doc_text(doc)))
    return [t for t in extract_keywords(query) if t in doc_terms]


# --------------------------------------------------------------------------- #
# Semantic search
# --------------------------------------------------------------------------- #
def semantic_search(
    query_vec,
    doc_vecs,
    documents: list[dict],
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    """Rank documents by COSINE SIMILARITY between embeddings.

    query_vec : embedding of the query (1-D)
    doc_vecs  : embeddings of the documents (n_docs x n_dims), same order as
                `documents` and produced by the SAME model as query_vec.
    """
    if len(documents) == 0:
        return []
    if len(documents) != len(doc_vecs):
        raise ValueError("documents and doc_vecs must have the same length.")
    scores = cosine_similarity_matrix(query_vec, doc_vecs)
    results = [
        {"index": i, "doc": documents[i], "score": float(scores[i])}
        for i in rank_by_score(scores, top_k=top_k)
    ]
    if min_score is not None:
        results = [r for r in results if r["score"] >= min_score]
    return results


def run_semantic_search(query: str, documents: list[dict], top_k: int = 5, min_score: float | None = None):
    """Embed the query and the documents, then rank. Returns (results, query_vec).

    Document embeddings come from a cache keyed on the document texts, so they
    are recomputed automatically whenever the collection changes.
    """
    from utils.model import embed_query, embed_texts  # lazy: needs Streamlit + model

    docs, _ = prepare_documents(documents)
    if not docs or not str(query or "").strip():
        return [], None
    doc_vecs = embed_texts(tuple(doc_text(d) for d in docs))
    query_vec = embed_query(str(query).strip())
    return semantic_search(query_vec, doc_vecs, docs, top_k=top_k, min_score=min_score), query_vec


# --------------------------------------------------------------------------- #
# RAG preview (string building only: no language model is called)
# --------------------------------------------------------------------------- #
def build_rag_prompt(question: str, retrieved: list[dict]) -> str:
    """Show what a RAG system would send to an LLM. We only build the text."""
    context = "\n\n".join(
        f"[{i}] {r['doc']['title']}: {r['doc']['text']}" for i, r in enumerate(retrieved, start=1)
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
