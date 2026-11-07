"""Embedding model loading and cached embedding helpers.

Two kinds of caching are used, on purpose:

* st.cache_resource  -> the model object itself (a big, shared, unpicklable
                        resource that must be created exactly once).
* st.cache_data      -> embedding arrays. The cache KEY is the tuple of texts,
                        so when the document collection changes the key
                        changes and fresh embeddings are computed. That is
                        automatic cache invalidation with no extra code.
"""
from __future__ import annotations

import re

import numpy as np
import streamlit as st

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_INPUT_CHARS = 2000  # keeps the UI and the model fast on very long pastes


@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads about 90 MB)...")
def load_model():
    """Load all-MiniLM-L6-v2 once per server process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def clean_text(text, max_chars: int = MAX_INPUT_CHARS) -> tuple[str, bool]:
    """Collapse whitespace, trim, and cap the length. Returns (text, was_truncated)."""
    if text is None:
        return "", False
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned[:max_chars], len(cleaned) > max_chars


def encode_texts(texts) -> np.ndarray:
    """Turn a list of texts into a (n_texts x 384) array of embeddings."""
    model = load_model()
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalise inside cosine similarity ourselves
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


@st.cache_data(show_spinner=False, max_entries=256)
def embed_texts(texts: tuple[str, ...]) -> np.ndarray:
    """Cached embeddings. Pass a TUPLE so Streamlit can hash it as the cache key."""
    if len(texts) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    return encode_texts(texts)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query with the same model used for documents."""
    return embed_texts((query,))[0]


def tokenize_for_model(text: str) -> list[str]:
    """The word-piece tokens the model actually sees (not the same as embeddings)."""
    return load_model().tokenizer.tokenize(text)


def model_max_tokens() -> int:
    """Longest input (in tokens) the model reads; anything longer is cut off."""
    return int(load_model().max_seq_length)
