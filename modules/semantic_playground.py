"""Module 4: Interactive Semantic Search Playground (10 min)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data import PLAYGROUND_CATEGORIES, PLAYGROUND_DOCS, PLAYGROUND_EXAMPLES, PLAYGROUND_QUERY
from utils.model import MAX_INPUT_CHARS, clean_text
from utils.search import (
    extract_keywords, is_valid_document, keyword_search, prepare_documents, run_semantic_search,
    shared_terms,
)
from utils.similarity import interpret_similarity
from utils.ui import (
    callout, knowledge_check, module_footer, module_header, result_card, section, takeaway,
)

PAGE_ID = "m4"
TOP_K = 5
MAX_CUSTOM_DOCS = 20
LOW_SCORE = 0.25


# ----------------------------------------------------------------------------- state
def _custom_docs() -> list[dict]:
    return st.session_state.setdefault("pg_custom_docs", [])


def _all_docs() -> list[dict]:
    docs, _ = prepare_documents(PLAYGROUND_DOCS + _custom_docs())
    return docs


def _set_query(q: str) -> None:
    st.session_state["pg_query"] = q


def _remove_custom(index: int) -> None:
    docs = _custom_docs()
    if 0 <= index < len(docs):
        docs.pop(index)


def _clear_custom() -> None:
    st.session_state["pg_custom_docs"] = []


def _add_custom(title: str, category: str, text: str) -> None:
    cleaned, cut = clean_text(text)
    doc = {"title": (title or "").strip() or "My document", "category": category, "text": cleaned, "custom": True}
    docs = _custom_docs()
    if len(docs) >= MAX_CUSTOM_DOCS:
        st.session_state["pg_form_msg"] = ("error", f"You can add up to {MAX_CUSTOM_DOCS} documents.")
    elif not is_valid_document(doc):
        st.session_state["pg_form_msg"] = ("error", "A document needs at least 10 characters of real text (letters, not just symbols).")
    elif any(d["text"].lower() == cleaned.lower() for d in PLAYGROUND_DOCS + docs):
        st.session_state["pg_form_msg"] = ("warning", "That document is already in the collection.")
    else:
        docs.append(doc)
        extra = f" (shortened to {MAX_INPUT_CHARS:,} characters)" if cut else ""
        st.session_state["pg_form_msg"] = ("success", f"Added “{doc['title']}”{extra}. Search again to see it ranked.")


# ----------------------------------------------------------------------------- page
def render() -> None:
    module_header(
        4,
        "Interactive Semantic Search Playground",
        10,
        "Run keyword and semantic search side by side and explain why their results differ.",
    )

    section("1 · The simple explanation", "📖")
    st.markdown(
        """
Both search types take a query and **rank** documents. They differ in *how* they score:

| | Keyword search | Semantic search |
|---|---|---|
| **Looks at** | Exact words | Meaning (embeddings) |
| **Score** | Share of query words found (0–100%) | **Cosine similarity** between vectors (−1 to 1) |
| **Returns** | Only documents sharing a word | Always the closest documents, even if weakly related |

Semantic search works in 4 steps: **embed the query → compare with stored document embeddings → compute
cosine similarity → rank and show the top 5.**
"""
    )
    section("2 · A real-world analogy", "🧩")
    callout(
        "analogy",
        "Finding a song",
        "Searching a music app for <b>“something calm to study to”</b>. A keyword search looks for songs with those "
        "words in the title. A semantic search understands the <i>mood</i> you describe and suggests lo-fi playlists "
        "that never mention “calm” or “study”.",
    )
    section("3 · A practical example", "🎓")
    st.markdown(
        "Try the first suggested query below, *“How do I make my laptop run faster?”*. The best document, "
        "“Speeding up a sluggish computer”, never uses the word *laptop*, *make* or *faster*. Watch what each mode does."
    )

    section("4 · Try it yourself", "🕹️")
    _custom_docs()
    st.session_state.setdefault("pg_query", PLAYGROUND_QUERY)
    st.markdown("**Suggested queries**")
    ex_cols = st.columns(3)
    for i, q in enumerate(PLAYGROUND_EXAMPLES):
        with ex_cols[i % 3]:
            st.button(q, key=f"pg_ex_{i}", on_click=_set_query, args=(q,))

    query = st.text_input("Your search query (press Enter to search)", key="pg_query", max_chars=300)
    mode = st.radio(
        "Search mode",
        ["Keyword search", "Semantic search", "Compare side by side"],
        index=2, horizontal=True, key="pg_mode",
    )

    _document_manager()

    docs = _all_docs()
    query_clean, _ = clean_text(query, 300)
    if not query_clean:
        st.info("Enter a query (or click a suggestion) to search the collection.")
    elif not docs:
        st.warning("The collection has no valid documents to search.")
    else:
        kw_results = keyword_search(query_clean, docs, top_k=TOP_K) if mode != "Semantic search" else []
        sem_results = []
        if mode != "Keyword search":
            with st.spinner("Embedding the query and comparing with document embeddings..."):
                sem_results, _ = run_semantic_search(query_clean, docs, top_k=TOP_K)

        if mode == "Keyword search":
            _render_keyword(query_clean, kw_results)
        elif mode == "Semantic search":
            _render_semantic(query_clean, sem_results)
        else:
            left, right = st.columns(2)
            with left:
                _render_keyword(query_clean, kw_results)
            with right:
                _render_semantic(query_clean, sem_results)
        _explain(query_clean, mode, kw_results, sem_results)

    knowledge_check(
        "m4",
        "In the playground, a semantic result has a cosine score of 0.12. What is the best interpretation?",
        [
            "The document is 12% correct",
            "It is the closest document available but is probably not very relevant",
            "The document contains 12% of your query words",
            "The search engine has crashed",
        ],
        correct=1,
        explanations=[
            "Cosine similarity isn't a percentage of correctness. High scores don't prove a document is factually right either.",
            "Right. Semantic search always ranks something first, so a low top score means 'nothing close exists'.",
            "That describes the keyword score, a completely different measure.",
            "Low scores are normal when no document matches the topic.",
        ],
    )
    takeaway(
        "Keyword scores count <b>shared words</b>; semantic scores measure <b>closeness of meaning</b> using cosine "
        "similarity. A high semantic score means “similar in meaning to what the model learned”, <b>not</b> “factually correct”."
    )
    module_footer(PAGE_ID)


# ----------------------------------------------------------------------------- pieces
def _document_manager() -> None:
    custom = _custom_docs()
    with st.expander(f"📚 Document collection ({len(PLAYGROUND_DOCS)} built-in + {len(custom)} yours)  ·  add your own"):
        table = pd.DataFrame(
            [{"Source": "yours" if d.get("custom") else "built-in", "Category": d["category"], "Title": d["title"],
              "Text": d["text"][:90] + ("…" if len(d["text"]) > 90 else "")} for d in PLAYGROUND_DOCS + custom]
        )
        st.dataframe(table, hide_index=True)

        st.markdown("**Add your own document**")
        with st.form("pg_add_form", clear_on_submit=True):
            title = st.text_input("Title", max_chars=80)
            category = st.selectbox("Category", PLAYGROUND_CATEGORIES + ["Other"])
            text = st.text_area("Document text", height=110, help="Write 1–3 sentences. Very long text is shortened to 2,000 characters.")
            submitted = st.form_submit_button("Add to collection")
        if submitted:
            _add_custom(title, category, text)
        msg = st.session_state.pop("pg_form_msg", None)
        if msg:
            getattr(st, msg[0])(msg[1])

        if custom:
            st.markdown("**Your documents**")
            for i, d in enumerate(custom):
                c1, c2 = st.columns([6, 1])
                c1.write(f"**{d['title']}** · {d['category']} — {d['text'][:80]}{'…' if len(d['text']) > 80 else ''}")
                c2.button("Remove", key=f"pg_rm_{i}", on_click=_remove_custom, args=(i,))
            st.button("Remove all my documents", key="pg_clear", on_click=_clear_custom)
        st.caption("Embeddings are cached against the exact document texts. Adding or removing a document changes "
                   "the cache key, so the index is rebuilt automatically.")


def _render_keyword(query: str, results: list[dict]) -> None:
    st.markdown("##### 🟠 Keyword search")
    terms = extract_keywords(query)
    if not terms:
        st.warning("Your query has only filler words (like *the, how, my*), so keyword search has nothing to match.")
        return
    st.caption(f"Keywords searched: **{', '.join(terms)}**")
    if not results:
        st.error("**0 results.** No document contains any of your keywords.")
        return
    for rank, r in enumerate(results, start=1):
        result_card(rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "kw",
                    note="Matched: " + ", ".join(r["matched_terms"]), highlight=r["matched_terms"])
    st.caption("🟠 **Keyword-match score** = share of query words found. It is *not* an embedding similarity.")


def _render_semantic(query: str, results: list[dict]) -> None:
    st.markdown("##### 🟣 Semantic search")
    if not results:
        st.warning("No semantic results (empty query or no valid documents).")
        return
    for rank, r in enumerate(results, start=1):
        label, _ = interpret_similarity(r["score"])
        result_card(rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "sem",
                    note=label, highlight=shared_terms(query, r["doc"]))
    st.caption("🟣 **Cosine similarity** between the query embedding and each document embedding (all-MiniLM-L6-v2).")


def _explain(query: str, mode: str, kw: list[dict], sem: list[dict]) -> None:
    section("Why these results?", "🔍")
    lines: list[str] = []
    if mode == "Keyword search":
        if kw:
            top = kw[0]
            lines.append(f"**“{top['doc']['title']}”** ranked first because it contains {len(top['matched_terms'])} of your "
                         f"{len(extract_keywords(query))} keywords ({', '.join(top['matched_terms'])}). Keyword search "
                         "doesn't check whether the document actually answers your question.")
        else:
            lines.append("Nothing matched because no document shares a word with your query. A document could still mean "
                         "exactly what you want. Switch to **Semantic search** to see.")
    else:
        if sem:
            top = sem[0]
            shared = shared_terms(query, top["doc"])
            if shared:
                lines.append(f"**“{top['doc']['title']}”** ranked first (cosine {top['score']:.3f}). It also shares the word(s) "
                             f"{', '.join(shared)} with your query, but the score comes from the *whole meaning*, not those words.")
            else:
                lines.append(f"**“{top['doc']['title']}”** ranked first (cosine {top['score']:.3f}) even though it shares **no keywords** "
                             "with your query. Its meaning is close to your question's, which is the strength of semantic search.")
            if top["score"] < LOW_SCORE:
                lines.append("⚠️ The best score is low. Semantic search always returns the *closest* documents, even when none are "
                             "truly relevant, so a low top score is a hint that the collection may not contain an answer.")
    if mode == "Compare side by side" and sem:
        kw_titles = {r["doc"]["title"] for r in kw}
        only_sem = [r["doc"]["title"] for r in sem if r["doc"]["title"] not in kw_titles]
        if only_sem:
            lines.append("Found by **semantic only**: " + "; ".join(f"“{t}”" for t in only_sem[:3]) + ". "
                         "Keyword search couldn't retrieve these because the wording differs.")
        elif kw:
            lines.append("Here both methods found the same documents, usually because the query reuses the documents' own words.")
    for line in lines:
        st.markdown("- " + line)
    callout(
        "warn",
        "Remember",
        "A similarity score shows that texts are <i>close in the model's learned space</i>. It does <b>not</b> prove a "
        "document is correct, and the model doesn't understand text like a person. Embedding search only <b>retrieves</b>; "
        "it never writes answers.",
    )
