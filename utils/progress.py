"""Page registry, navigation and completion tracking (all in st.session_state)."""
from __future__ import annotations

import streamlit as st

MODULES = [
    {"id": "m1", "num": 1, "icon": "🔎", "short": "The Problem", "minutes": 4,
     "title": "The Problem: Why Keyword Search Is Not Enough",
     "blurb": "See keyword search fail on a simple question."},
    {"id": "m2", "num": 2, "icon": "🧬", "short": "Embeddings", "minutes": 8,
     "title": "Understanding Embeddings",
     "blurb": "Turn sentences into real vectors with a neural model."},
    {"id": "m3", "num": 3, "icon": "🧭", "short": "Vector Space", "minutes": 7,
     "title": "Vector Space and Similarity",
     "blurb": "Measure closeness with cosine similarity and map it with PCA."},
    {"id": "m4", "num": 4, "icon": "🧪", "short": "Playground", "minutes": 10,
     "title": "Semantic Search Playground",
     "blurb": "Compare keyword and semantic search on your own queries."},
    {"id": "m5", "num": 5, "icon": "🏗️", "short": "Mini Project", "minutes": 10,
     "title": "Build a Mini Semantic Search Application",
     "blurb": "Build a university knowledge base search engine."},
    {"id": "m6", "num": 6, "icon": "🏁", "short": "Quiz & Challenge", "minutes": 6,
     "title": "Knowledge Check and Final Challenge",
     "blurb": "Test yourself, then complete a practical challenge."},
]
HOME = {"id": "home", "num": 0, "icon": "🏠", "short": "Home", "minutes": 0, "title": "Home", "blurb": ""}
PAGES = [HOME] + MODULES
PAGE_IDS = [p["id"] for p in PAGES]
TOTAL_MINUTES = sum(m["minutes"] for m in MODULES)

# Session-state key prefixes owned by the modules (cleared by "Reset progress")
_RESETTABLE_PREFIXES = ("kc_", "quiz_", "emb_", "vs_", "pg_", "kb_", "ch_", "m1_", "completed")


def page_by_id(page_id: str) -> dict:
    return next(p for p in PAGES if p["id"] == page_id)


def init_state() -> None:
    st.session_state.setdefault("nav", "home")
    st.session_state.setdefault("completed", set())


def go_to(page_id: str) -> None:
    st.session_state["nav"] = page_id


def is_complete(page_id: str) -> bool:
    return page_id in st.session_state.get("completed", set())


def mark_complete(page_id: str) -> None:
    st.session_state.setdefault("completed", set()).add(page_id)


def complete_and_go(page_id: str, next_id: str) -> None:
    mark_complete(page_id)
    go_to(next_id)


def completed_count() -> int:
    return sum(1 for m in MODULES if is_complete(m["id"]))


def minutes_left() -> int:
    return sum(m["minutes"] for m in MODULES if not is_complete(m["id"]))


def reset_progress() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(_RESETTABLE_PREFIXES):
            del st.session_state[key]
    st.session_state["completed"] = set()
    st.session_state["nav"] = "home"
