"""BEYOND KEYWORDS: Understanding Embeddings & Semantic Search.

Run with:  streamlit run app.py
"""
import streamlit as st

# set_page_config must be the first Streamlit call
st.set_page_config(
    page_title="Beyond Keywords · Embeddings & Semantic Search",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules import embeddings, home, introduction, mini_project, quiz, semantic_playground, vector_space  # noqa: E402
from utils.progress import (  # noqa: E402
    MODULES, PAGE_IDS, TOTAL_MINUTES, completed_count, init_state, is_complete, minutes_left,
    page_by_id, reset_progress,
)
from utils.ui import inject_css  # noqa: E402

ROUTES = {
    "home": home.render,
    "m1": introduction.render,
    "m2": embeddings.render,
    "m3": vector_space.render,
    "m4": semantic_playground.render,
    "m5": mini_project.render,
    "m6": quiz.render,
}


def _label(page_id: str) -> str:
    page = page_by_id(page_id)
    if page_id == "home":
        return f"{page['icon']}  Home"
    tick = "✅" if is_complete(page_id) else "○"
    return f"{tick}  {page['num']}. {page['short']} · {page['minutes']} min"


def main() -> None:
    init_state()
    inject_css()

    with st.sidebar:
        st.markdown("## 🧭 Beyond Keywords")
        st.caption("Embeddings & Semantic Search")
        done = completed_count()
        st.progress(done / len(MODULES), text=f"{done} of {len(MODULES)} modules complete")
        st.caption(f"⏱ About {minutes_left()} of {TOTAL_MINUTES} minutes left")
        st.radio("Navigate", PAGE_IDS, format_func=_label, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("Model: `all-MiniLM-L6-v2` · cosine similarity")
        st.button("Reset my progress", key="reset_progress", on_click=reset_progress)

    ROUTES[st.session_state["nav"]]()


main()
