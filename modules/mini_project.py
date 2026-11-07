"""Module 5: Build a Mini Semantic Search Application (10 min).

Project: "University Knowledge Base Search Engine". This is a RETRIEVAL system:
it finds and ranks existing documents. It does not generate answers.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import utils.model as model_utils
import utils.search as search_utils
import utils.similarity as sim_utils
from utils.data import UNIVERSITY_DOCS, UNIVERSITY_EXAMPLES
from utils.model import MAX_INPUT_CHARS, clean_text, embed_query, embed_texts
from utils.search import (
    build_rag_prompt, doc_text, extract_keywords, is_valid_document, keyword_search,
    semantic_search, shared_terms,
)
from utils.ui import (
    BLUE, VIOLET, callout, flow, knowledge_check, module_footer, module_header, result_card,
    section, style_figure, takeaway,
)

PAGE_ID = "m5"
MAX_DOCS = 30

_LOAD_MODEL_SNIPPET = '''@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
'''


# ----------------------------------------------------------------------------- state
def _docs() -> list[dict]:
    if "kb_docs" not in st.session_state:
        st.session_state["kb_docs"] = [dict(d, mine=False) for d in UNIVERSITY_DOCS]
    return st.session_state["kb_docs"]


def _valid_docs() -> list[dict]:
    return [d for d in _docs() if is_valid_document(d)]


def _texts() -> tuple[str, ...]:
    return tuple(doc_text(d) for d in _valid_docs())


def _index_is_current() -> bool:
    idx = st.session_state.get("kb_index")
    return bool(idx) and idx["texts"] == _texts()


def _set_question(q: str) -> None:
    st.session_state["kb_question"] = q


def _remove_doc(i: int) -> None:
    docs = _docs()
    if 0 <= i < len(docs):
        docs.pop(i)


def _reset_samples() -> None:
    st.session_state["kb_docs"] = [dict(d, mine=False) for d in UNIVERSITY_DOCS]
    st.session_state.pop("kb_last", None)


def _clear_docs() -> None:
    st.session_state["kb_docs"] = []
    st.session_state.pop("kb_last", None)


# ----------------------------------------------------------------------------- page
def render() -> None:
    module_header(
        5,
        "Build a Mini Semantic Search Application",
        10,
        "Apply embeddings and cosine similarity to build a working university knowledge base search engine.",
    )

    section("1 · The simple explanation", "📖")
    st.markdown(
        """
You will build a small **semantic retrieval** system. Students type a question in everyday language and the
app returns the **most relevant policy documents**, ranked by cosine similarity.

> **Important:** this app **retrieves** documents; it does **not** generate answers. No language model (LLM) is used.
> It points to *where* the answer is, and you read the document yourself.
"""
    )
    section("2 · A real-world analogy", "🧩")
    callout(
        "analogy",
        "A smart noticeboard",
        "Imagine a university noticeboard that, when you ask <i>“Can I take the exam if I missed many classes?”</i>, "
        "instantly pulls out the <b>Attendance Policy</b> sheet. It doesn't write a reply; it just hands you the "
        "right paper. That's retrieval.",
    )
    section("3 · A practical example: the architecture", "🎓")
    flow([("User query", ""), ("Embedding model", "hl"), ("Query vector", ""), ("Similarity calculation", "hl"),
          ("Ranked documents", ""), ("Retrieved results", "")])
    st.caption("Documents are embedded **once** (the index). Each question is embedded at search time and compared with that index, using the same model.")

    section("4 · Try it yourself", "🕹️")
    tab_build, tab_search, tab_code, tab_rag = st.tabs(
        ["① Knowledge base & index", "② Ask questions", "③ Code walkthrough", "④ Towards RAG"]
    )
    with tab_build:
        _tab_build()
    with tab_search:
        _tab_search()
    with tab_code:
        _tab_code()
    with tab_rag:
        _tab_rag()

    knowledge_check(
        "m5",
        "Which statement about this Module 5 application is true?",
        [
            "It uses an LLM to write answers from the documents",
            "It embeds documents once, embeds each question, and ranks documents by cosine similarity",
            "It searches by counting shared keywords",
            "It trains a new model every time you add a document",
        ],
        correct=1,
        explanations=[
            "No LLM is involved. The app only retrieves and ranks existing documents.",
            "Correct. That's the retrieval pipeline: index once, embed the query, compare, rank.",
            "That is keyword search, which we only use for comparison.",
            "The model is pre-trained and never changes. We only compute new embeddings for new text.",
        ],
    )
    takeaway(
        "A semantic search app is: <b>embed documents once → embed the query → cosine similarity → rank</b>. It "
        "<b>retrieves</b> text; a RAG system adds an LLM <i>after</i> this step to write an answer from what was retrieved."
    )
    module_footer(PAGE_ID)


# ----------------------------------------------------------------------------- tabs
def _tab_build() -> None:
    docs = _docs()
    st.markdown("**Step 1: your knowledge base.** These are fictional sample policies. Add, remove or replace them.")
    for i, d in enumerate(docs):
        c1, c2 = st.columns([7, 1])
        tag = " ✍️" if d.get("mine") else ""
        flag = "" if is_valid_document(d) else " ⚠️ *(too short, will be ignored)*"
        c1.markdown(f"**{d['title']}**{tag}{flag} · {d['text'][:110]}{'…' if len(d['text']) > 110 else ''}")
        c2.button("Remove", key=f"kb_rm_{i}", on_click=_remove_doc, args=(i,))
    b1, b2, _ = st.columns([1.3, 1.3, 3])
    b1.button("Reset to samples", key="kb_reset", on_click=_reset_samples)
    b2.button("Clear all", key="kb_clear", on_click=_clear_docs)

    with st.form("kb_add_form", clear_on_submit=True):
        st.markdown("**Add a university document**")
        title = st.text_input("Title (e.g. Hostel curfew)", max_chars=80)
        text = st.text_area("Document text", height=100)
        add = st.form_submit_button("Add document")
    if add:
        cleaned, cut = clean_text(text)
        new_doc = {"title": title.strip() or "Untitled", "category": "Student", "text": cleaned, "mine": True}
        if len(docs) >= MAX_DOCS:
            st.error(f"The knowledge base is limited to {MAX_DOCS} documents.")
        elif not is_valid_document(new_doc):
            st.error("Please write at least 10 characters of real text.")
        elif any(d["text"].lower() == cleaned.lower() for d in docs):
            st.warning("That document is already in the knowledge base.")
        else:
            docs.append(new_doc)
            st.success(f"Added “{new_doc['title']}”." + (f" Text shortened to {MAX_INPUT_CHARS:,} characters." if cut else ""))
            st.rerun()

    st.markdown("**Step 2: generate embeddings (build the index).**")
    valid = _valid_docs()
    if not valid:
        st.warning("Add at least one document with real text first.")
        return
    if st.button("⚙️ Generate embeddings", type="primary", key="kb_build"):
        with st.spinner("Embedding every document with all-MiniLM-L6-v2..."):
            start = time.perf_counter()
            vectors = embed_texts(_texts())
            elapsed = time.perf_counter() - start
        st.session_state["kb_index"] = {"texts": _texts(), "vectors": vectors, "seconds": elapsed}
        st.rerun()

    idx = st.session_state.get("kb_index")
    if idx and _index_is_current():
        vectors = idx["vectors"]
        st.success(f"Index ready: **{vectors.shape[0]} documents × {vectors.shape[1]} dimensions** "
                   f"(computed in {idx['seconds']:.2f}s; repeat builds are instant thanks to caching).")
        preview = pd.DataFrame(
            np.round(vectors[:, :6], 3),
            columns=[f"dim {i}" for i in range(6)],
            index=[d["title"] for d in valid],
        )
        with st.expander("Peek at the stored vectors (first 6 of 384 dimensions)"):
            st.dataframe(preview)
            st.caption("Each row is one document's embedding. This table of numbers is the whole 'index'.")
    elif idx:
        st.warning("The documents changed since the index was built. Press **Generate embeddings** again to refresh it.")
    else:
        st.info("No index yet. Press **Generate embeddings** to build it.")


def _tab_search() -> None:
    if not _index_is_current():
        st.warning("Build (or rebuild) the index in tab ① first. Searching needs up-to-date document embeddings.")
        return
    st.session_state.setdefault("kb_question", UNIVERSITY_EXAMPLES[0])
    st.markdown("**Suggested questions**")
    cols = st.columns(3)
    for i, q in enumerate(UNIVERSITY_EXAMPLES):
        with cols[i % 3]:
            st.button(q, key=f"kb_ex_{i}", on_click=_set_question, args=(q,))

    with st.form("kb_search_form"):
        st.text_input("Ask a question in your own words", key="kb_question", max_chars=300)
        c1, c2, c3 = st.columns(3)
        top_k = c1.slider("Results to show", 1, 5, 3)
        threshold = c2.slider("Minimum similarity", 0.0, 0.6, 0.0, 0.05,
                              help="Hide results scoring below this cosine similarity.")
        compare = c3.checkbox("Also show keyword search", value=True)
        go_search = st.form_submit_button("Search", type="primary")
    if go_search:
        question, _ = clean_text(st.session_state["kb_question"], 300)
        if not question:
            st.warning("Please type a question first.")
            st.session_state.pop("kb_last", None)
        else:
            st.session_state["kb_last"] = {"q": question, "k": top_k, "t": threshold, "cmp": compare}
            _log_query(question, compare)

    last = st.session_state.get("kb_last")
    if not last:
        st.info("Press **Search** to retrieve documents.")
        return

    idx = st.session_state["kb_index"]
    docs = _valid_docs()
    with st.spinner("Embedding the question and ranking documents..."):
        qvec = embed_query(last["q"])
        results = semantic_search(qvec, idx["vectors"], docs, top_k=last["k"], min_score=last["t"] or None)

    st.markdown(f"#### Results for: *{last['q']}*")
    if last["cmp"]:
        left, right = st.columns(2)
    else:
        left, right = st.container(), None
    with left:
        st.markdown("##### 🟣 Semantic retrieval")
        if not results:
            st.warning("No document reached your minimum similarity. Lower the threshold or add a relevant document.")
        for rank, r in enumerate(results, start=1):
            result_card(rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "sem",
                        highlight=shared_terms(last["q"], r["doc"]))
        st.caption("Scores are cosine similarities. Retrieval only: no answer is generated.")
    if right is not None:
        with right:
            st.markdown("##### 🟠 Keyword search")
            kw = keyword_search(last["q"], docs, top_k=last["k"])
            if not extract_keywords(last["q"]):
                st.warning("Only filler words in the question.")
            elif not kw:
                st.error("0 results: no document shares a word with this question.")
            for rank, r in enumerate(kw, start=1):
                result_card(rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "kw",
                            note="Matched: " + ", ".join(r["matched_terms"]), highlight=r["matched_terms"])
            st.caption("Orange = keyword-match score, not an embedding similarity.")

    with st.expander("🔬 Retrieval trace: see what happened inside"):
        all_scores = sim_utils.cosine_similarity_matrix(qvec, idx["vectors"])
        st.markdown(
            f"1. The question became a **query vector** of {qvec.shape[0]} numbers (first 5: "
            f"`{', '.join(f'{x:.3f}' for x in qvec[:5])}`).\n"
            f"2. It was compared with all **{len(docs)}** document vectors using cosine similarity.\n"
            "3. The scores were sorted from highest to lowest (below), and the top results were shown."
        )
        order = sim_utils.rank_by_score(all_scores, top_k=len(all_scores))
        fig = go.Figure(go.Bar(
            x=[float(all_scores[i]) for i in order][::-1],
            y=[docs[i]["title"] for i in order][::-1],
            orientation="h", marker_color=[VIOLET if rnk < last["k"] else BLUE for rnk in range(len(order))][::-1],
        ))
        fig.update_layout(title="Cosine similarity of the question to every document", xaxis_title="Cosine similarity")
        st.plotly_chart(style_figure(fig, max(260, 34 * len(order) + 90)))
        st.caption("Similarity shows closeness of meaning in the model's learned space. It does not guarantee the document is correct or up to date.")


def _log_query(question: str, compared: bool) -> None:
    log = st.session_state.setdefault("kb_query_log", [])
    log.append({"query": question, "compared": compared})


def _tab_code() -> None:
    st.markdown("These are the **actual functions** running this app (read live from the source files).")
    items = [
        ("1. Load the model once", _LOAD_MODEL_SNIPPET,
         "`@st.cache_resource` makes Streamlit load the neural network a single time and reuse it for every user and rerun. "
         "(Simplified copy of `load_model` in `utils/model.py`.)"),
        ("2. Turn text into vectors", model_utils.encode_texts,
         "`model.encode(...)` runs the neural network and returns one 384-number vector per text. `embed_texts` wraps this with `@st.cache_data`, "
         "so identical documents are never embedded twice and changed documents get fresh vectors."),
        ("3. Compare vectors", sim_utils.cosine_similarity_matrix,
         "Computes `A · B / (‖A‖ × ‖B‖)` for the query against every document at once using matrix maths."),
        ("4. Sort the scores", sim_utils.rank_by_score,
         "`argsort` on the negative scores gives document positions from best to worst."),
        ("5. Put it together", search_utils.semantic_search,
         "Scores every document, ranks them, and returns the top results with their scores."),
    ]
    for title, fn, why in items:
        st.markdown(f"**{title}**")
        st.markdown(why)
        if isinstance(fn, str):
            source = fn
        else:
            try:
                source = inspect.getsource(fn)
            except (OSError, TypeError):
                source = "# (source not available in this environment)"
        st.code(source, language="python")


def _tab_rag() -> None:
    st.markdown(
        "**RAG** (Retrieval-Augmented Generation) adds a language model *after* retrieval. "
        "What you built is the **R** (retrieval). The generation step is **not** part of this app."
    )
    flow([("Question", ""), ("Retriever (this app)", "hl"), ("Top documents", "hl"), ("Prompt builder", "off"),
          ("LLM", "off"), ("Grounded answer", "off")])
    st.caption("Dashed boxes are not implemented here.")
    st.markdown(
        """
To turn this into RAG you would typically:
1. **Chunk** long documents into passages of a few hundred words.
2. Store chunk embeddings in a **vector database** (for example FAISS, Chroma or pgvector) so search stays fast at scale.
3. Retrieve the top chunks for each question (what this app does).
4. **Insert them into a prompt** and ask an LLM to answer using only that context.
5. Show the answer *with citations* to the retrieved documents, because retrieval can still return the wrong text.
"""
    )
    st.markdown("**Preview: the prompt a RAG system would send** (we only build the text; no LLM is called)")
    last = st.session_state.get("kb_last")
    if not last or not _index_is_current():
        st.info("Run a search in tab ② first, then come back to preview the prompt.")
    else:
        docs, idx = _valid_docs(), st.session_state["kb_index"]
        qvec = embed_query(last["q"])
        results = semantic_search(qvec, idx["vectors"], docs, top_k=min(3, last["k"]))
        st.code(build_rag_prompt(last["q"], results), language="text")
    try:
        st.code(inspect.getsource(build_rag_prompt), language="python")
    except (OSError, TypeError):
        pass
