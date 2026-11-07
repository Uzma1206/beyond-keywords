"""Module 1: The Problem - Why Keyword Search Is Not Enough (4 min)."""
from __future__ import annotations

import streamlit as st

from utils.data import DEMO_DOCS, DEMO_QUERIES, DEMO_QUERY
from utils.search import extract_keywords, keyword_search, run_semantic_search, shared_terms
from utils.ui import (
    callout, knowledge_check, module_footer, module_header, render_html,
    result_card, section, takeaway,
)

PAGE_ID = "m1"


def _set_query(q: str) -> None:
    st.session_state["m1_query"] = q


def render() -> None:
    module_header(
        1,
        "The Problem: Why Keyword Search Is Not Enough",
        4,
        "Explain why searching by exact words can miss documents that mean the same thing.",
    )

    # ------------------------------------------------------------------ Learn
    section("1 · The simple explanation", "📖")
    st.markdown(
        """
Most basic search tools work by **lexical matching**: they check whether the *letters and words*
of your query appear in a document. If the words match, the document is returned. If they don't,
it is invisible, even when it is exactly what you need.

**Semantic matching** is different. It compares the *meaning* of the query and the document, so
"speed up a slow PC" and "make my laptop faster" can be recognised as close, even though they
share almost no words.
"""
    )

    section("2 · A real-world analogy", "🧩")
    callout(
        "analogy",
        "The strict librarian vs. the helpful librarian",
        "Imagine asking a librarian for a book on <b>“making my laptop faster”</b>. A <b>strict librarian</b> "
        "only hands you books whose title contains those exact words. A <b>helpful librarian</b> understands "
        "you want to speed up a computer and brings you <i>“How to speed up a slow PC”</i>. "
        "Keyword search is the strict librarian. Semantic search aims to be the helpful one.",
    )

    section("3 · A practical example", "🎓")
    st.markdown(f"A student searches for: **“{DEMO_QUERY}”**. Our tiny library has four documents:")
    keywords = extract_keywords(DEMO_QUERY)
    rows = []
    for doc in DEMO_DOCS:
        hit = shared_terms(DEMO_QUERY, doc)
        rows.append({"Document": doc["title"], "Query keywords found": ", ".join(hit) if hit else "none"})
    st.caption(
        f"Keywords left after ignoring filler words like *how, can, I, my*: **{', '.join(keywords)}**"
    )
    st.dataframe(rows, hide_index=True)
    callout(
        "warn",
        "What went wrong?",
        "The two documents that actually answer the question (<i>“Tips to improve computer performance”</i> and "
        "<i>“How to speed up a slow PC”</i>) share <b>no</b> keywords with the query. The only match is the "
        "<i>battery</i> guide, which merely contains the word “laptop”. Keyword search found the wrong document "
        "and missed the right ones.",
    )

    # --------------------------------------------------------------- Activity
    section("4 · Try it yourself", "🕹️")
    st.markdown("Type your own query and see which documents plain keyword matching retrieves.")
    st.session_state.setdefault("m1_query", DEMO_QUERY)
    cols = st.columns(len(DEMO_QUERIES))
    for col, q in zip(cols, DEMO_QUERIES):
        with col:
            st.button(q, key=f"m1_ex_{q}", on_click=_set_query, args=(q,))
    query = st.text_input("Your search query", key="m1_query", max_chars=200)

    if not query.strip():
        st.info("Type a query above (or click an example) to run the keyword search.")
    else:
        results = keyword_search(query, DEMO_DOCS, top_k=4)
        q_terms = extract_keywords(query)
        if not q_terms:
            st.warning("Your query only contains filler words (like *the, of, how*), so there is nothing to match.")
        elif not results:
            st.error("**No documents retrieved.** None of your query words appear in any document, "
                     "even though some documents may mean the same thing.")
        else:
            st.markdown(f"Keyword search retrieved **{len(results)}** of {len(DEMO_DOCS)} documents:")
            for rank, r in enumerate(results, start=1):
                result_card(
                    rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "kw",
                    note=f"Matched words: {', '.join(r['matched_terms'])}", highlight=r["matched_terms"],
                )
            st.caption("Orange bars are **keyword-match scores** (the share of your query words found). "
                       "They are not similarity scores and say nothing about meaning.")

        with st.expander("🔭 Peek ahead: what would meaning-based search rank first? (uses the real embedding model)"):
            st.markdown(
                "You will learn how this works in Modules 2 and 3. For now, notice how the ranking changes. "
                "The first click downloads the model (about 90 MB) and may take a minute."
            )
            if st.button("Run semantic search on the same query", key="m1_peek"):
                with st.spinner("Embedding the query and documents..."):
                    sem, _ = run_semantic_search(query, DEMO_DOCS, top_k=4)
                st.session_state["m1_peek_results"] = (query, sem)
            peek = st.session_state.get("m1_peek_results")
            if peek and peek[0] == query:
                for rank, r in enumerate(peek[1], start=1):
                    result_card(rank, r["doc"]["title"], r["doc"]["category"], r["doc"]["text"], r["score"], "sem")
                st.caption("Violet bars are genuine **cosine similarities** between embeddings. "
                           "Documents about speeding up a computer should now rank above the battery guide.")

    section("Lexical vs. semantic matching", "⚖️")
    c1, c2 = st.columns(2)
    with c1:
        render_html(
            """<div class="bk-card"><span class="bk-chip amber">Lexical (keyword)</span>
            <div class="t">Matches words</div>
            <div class="d">✔ Fast, simple, exact<br>✔ Great for IDs, names, error codes<br>
            ✘ Misses synonyms (“fast” vs “quick”)<br>✘ Misses paraphrases and different wording<br>
            ✘ Can match the right word in the wrong context</div></div>"""
        )
    with c2:
        render_html(
            """<div class="bk-card"><span class="bk-chip violet">Semantic (embeddings)</span>
            <div class="t">Matches meaning</div>
            <div class="d">✔ Finds related ideas in different words<br>✔ Works with natural-language questions<br>
            ✘ Needs a trained model<br>✘ Can return loosely related results<br>
            ✘ Similarity is not proof of factual correctness</div></div>"""
        )

    # ---------------------------------------------------------- Knowledge check
    knowledge_check(
        "m1",
        "A student searches “speed up my laptop”. A document titled “Tips to improve computer performance” "
        "contains none of those words. What will pure keyword search do?",
        [
            "Return it, because it understands the topic",
            "Miss it, because no query word appears in the document",
            "Return it first, because it is the shortest document",
            "Translate the query until a word matches",
        ],
        correct=1,
        explanations=[
            "Keyword search has no understanding of topics. That ability belongs to semantic search.",
            "Exactly. Lexical matching needs overlapping words, and there are none here.",
            "Length plays no role in whether a keyword matches.",
            "Keyword search never rewrites or translates your query.",
        ],
    )

    takeaway(
        "Keyword search matches <b>words</b>; it fails when people use different words for the same idea. "
        "Semantic search aims to match <b>meaning</b>, and the rest of this course shows how."
    )
    module_footer(PAGE_ID)
