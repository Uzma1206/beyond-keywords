"""Module 6: Knowledge Check and Final Challenge (6 min)."""
from __future__ import annotations

import streamlit as st

from utils.progress import is_complete, mark_complete
from utils.ui import (
    callout, esc, module_footer, module_header, render_html, section, takeaway,
)

PAGE_ID = "m6"

# Each question: options, index of the right answer, and one explanation PER option.
QUESTIONS = [
    {
        "topic": "Embeddings",
        "q": "What is an embedding?",
        "options": [
            "A short written summary of a text",
            "A list of numbers that represents the meaning of a text",
            "A copy of the text stored in a database",
            "A list of the most frequent words in a text",
        ],
        "answer": 1,
        "why": [
            "Summaries are written text. An embedding is numeric.",
            "Correct. An embedding is a vector of numbers produced by a trained model.",
            "The text is stored separately. The embedding is a numeric representation of it.",
            "Word counts are a keyword technique, not an embedding.",
        ],
    },
    {
        "topic": "Vector representations",
        "q": "all-MiniLM-L6-v2 gives every text 384 numbers. What does '384' describe?",
        "options": [
            "The number of words in the text",
            "The number of tokens in the text",
            "The length of the embedding vector (its dimensions)",
            "The number of documents in the search index",
        ],
        "answer": 2,
        "why": [
            "The vector length is fixed no matter how many words the text has.",
            "Tokens are the input pieces. They vary per text, while 384 is constant.",
            "Yes. 384 is the number of dimensions of every embedding from this model.",
            "Index size depends on your documents. 384 is a property of the model.",
        ],
    },
    {
        "topic": "Semantic search",
        "q": "Why can semantic search match 'speed up my laptop' with 'tips to improve computer performance'?",
        "options": [
            "Both texts have the same number of letters",
            "It searches a built-in thesaurus for each word",
            "Their embeddings are close because the model learned they express similar ideas",
            "It translates both sentences into French first",
        ],
        "answer": 2,
        "why": [
            "Letter counts play no role.",
            "There is no thesaurus lookup. The relationship is learned during training.",
            "Right. Similar meaning gives similar vectors, so cosine similarity is high.",
            "Translation is not part of embedding search.",
        ],
    },
    {
        "topic": "Keyword search",
        "q": "Which query would keyword search handle WORST?",
        "options": [
            "Error code 0x80070057",
            "A product name copied from a catalogue",
            "A question that uses different words from the document, like 'cheap flights' vs 'budget airfare'",
            "A person's exact surname",
        ],
        "answer": 2,
        "why": [
            "Exact codes are a strength of keyword search.",
            "Exact names match perfectly with keywords.",
            "Correct. Different wording means no shared words, so lexical matching fails.",
            "Exact names are a strength of keyword search.",
        ],
    },
    {
        "topic": "Cosine similarity",
        "q": "What does cosine similarity compare?",
        "options": [
            "The direction (angle) of two vectors",
            "The number of identical words",
            "The length of two vectors only",
            "The file size of two documents",
        ],
        "answer": 0,
        "why": [
            "Yes. It divides out the lengths, so it compares only direction.",
            "Counting words is keyword matching, not cosine similarity.",
            "Dividing by the lengths removes length from the comparison.",
            "File size is unrelated.",
        ],
    },
    {
        "topic": "Cosine similarity",
        "q": "A query and a document have cosine similarity 0.05. What is the best conclusion?",
        "options": [
            "The document is 5% correct",
            "They are very similar",
            "They share almost no meaning, so the document is probably not relevant",
            "The embedding model failed",
        ],
        "answer": 2,
        "why": [
            "A cosine score isn't a percentage of correctness.",
            "Very similar texts score high, not near zero.",
            "Right. A value near 0 means the vectors are nearly at a right angle: little shared meaning.",
            "A low score is a normal, valid result for unrelated texts.",
        ],
    },
    {
        "topic": "Vector databases",
        "q": "What is the main job of a vector database in a search system?",
        "options": [
            "To write answers to questions",
            "To store embeddings and quickly find the nearest ones to a query vector",
            "To translate documents between languages",
            "To check whether documents are true",
        ],
        "answer": 1,
        "why": [
            "Writing answers is done by language models, not by the database.",
            "Yes. It stores vectors and does fast nearest-neighbour search, which matters when you have millions of documents.",
            "Translation is not what a vector database does.",
            "Vector databases can't verify facts. They only find similar vectors.",
        ],
    },
    {
        "topic": "Retrieval",
        "q": "Our University Knowledge Base app is best described as…",
        "options": [
            "A chatbot that writes new answers",
            "A semantic retrieval system that ranks existing documents",
            "A keyword counter",
            "A model that is retrained for each query",
        ],
        "answer": 1,
        "why": [
            "It doesn't generate text. No LLM is used.",
            "Correct. It retrieves and ranks stored documents by embedding similarity.",
            "Keyword counting is only used in the comparison panel.",
            "The embedding model is fixed and never retrained.",
        ],
    },
    {
        "topic": "Real-world applications",
        "q": "In which situation might keyword search still beat semantic search?",
        "options": [
            "Finding order number A-10293 in a database",
            "Finding articles about 'feeling anxious before exams' that use other words",
            "Recommending similar movies",
            "Matching a customer question to FAQ answers written in different words",
        ],
        "answer": 0,
        "why": [
            "Yes. Exact identifiers must match character for character. Embeddings could mix similar-looking codes.",
            "This needs meaning, so semantic search is the better fit.",
            "Recommendations rely on similarity of meaning or taste.",
            "FAQ matching is a classic semantic search use case.",
        ],
    },
    {
        "topic": "Limits of embeddings",
        "q": "A document gets the highest similarity score for your query. What does this guarantee?",
        "options": [
            "The document is factually correct",
            "The document contains your query words",
            "Nothing about correctness: it is only the closest match in the model's learned space",
            "The document is up to date",
        ],
        "answer": 2,
        "why": [
            "Similarity measures closeness of meaning, not truth.",
            "Semantic search doesn't need shared words, so this isn't guaranteed.",
            "Exactly. Always verify retrieved information, and remember embeddings approximate meaning.",
            "Similarity says nothing about whether text is current.",
        ],
    },
]

CHECKLIST = [
    "I added at least 5 university documents (they are meaningful, not placeholders).",
    "I built the index (generated embeddings) after my last change to the documents.",
    "I tested at least 3 different questions.",
    "At least one question used words that do NOT appear in the best document.",
    "I compared keyword and semantic results for each question.",
    "I explained why the results differed using the terms 'lexical' and 'semantic'.",
    "I interpreted at least one cosine score (high or low) correctly.",
    "I mentioned a limitation (similarity is not factual correctness, or low scores still return results).",
]


# ----------------------------------------------------------------------------- state
def _reset_quiz() -> None:
    for key in [k for k in st.session_state if k.startswith("quiz_")]:
        del st.session_state[key]


def _next() -> None:
    st.session_state["quiz_pos"] = st.session_state.get("quiz_pos", 0) + 1


# ----------------------------------------------------------------------------- page
def render() -> None:
    module_header(
        6,
        "Knowledge Check and Final Challenge",
        6,
        "Check your understanding with a quiz, then complete a practical semantic search challenge.",
    )
    section("1 · Learn, practise, prove it", "📖")
    st.markdown(
        "You have seen the **problem** (Module 1), the **representation** (2), the **maths** (3), the "
        "**experiment** (4) and the **application** (5). Now test yourself, then apply it."
    )
    callout(
        "analogy",
        "Like a driving test",
        "First the theory test (quiz), then the road test (the final challenge), where you do the real thing rather "
        "than only answer questions about it.",
    )

    tab_quiz, tab_challenge = st.tabs(["📝 Quiz (10 questions)", "🏁 Final challenge"])
    with tab_quiz:
        _quiz()
    with tab_challenge:
        _challenge()

    takeaway(
        "You can now explain <b>why</b> keyword search falls short, <b>how</b> embeddings and cosine similarity find "
        "meaning, and <b>how to build</b> a retrieval system, plus its limits: similarity is not truth, and retrieval is not generation."
    )
    module_footer(PAGE_ID)


# ----------------------------------------------------------------------------- quiz
def _quiz() -> None:
    st.session_state.setdefault("quiz_pos", 0)
    st.session_state.setdefault("quiz_answers", {})
    pos, answers, total = st.session_state["quiz_pos"], st.session_state["quiz_answers"], len(QUESTIONS)

    if pos >= total:
        _quiz_results()
        return

    q = QUESTIONS[pos]
    st.progress(pos / total, text=f"Question {pos + 1} of {total} · {q['topic']}")
    st.markdown(f"### {q['q']}")

    if pos not in answers:
        choice = st.radio("Choose one answer", list(range(len(q["options"]))),
                          format_func=lambda i: q["options"][i], index=None, key=f"quiz_radio_{pos}",
                          label_visibility="collapsed")
        if st.button("Submit answer", type="primary", key=f"quiz_submit_{pos}"):
            if choice is None:
                st.warning("Pick an answer first.")
            else:
                answers[pos] = choice
                st.rerun()
        return

    picked = answers[pos]
    for i, opt in enumerate(q["options"]):
        cls = "right" if i == q["answer"] else ("wrong" if i == picked else "")
        mark = "✅ " if i == q["answer"] else ("❌ " if i == picked else "")
        render_html(f'<div class="bk-opt {cls}">{mark}{esc(opt)}</div>')
    if picked == q["answer"]:
        st.success(f"**Correct!** {q['why'][picked]}")
    else:
        st.error(f"**Not quite.** {q['why'][picked]}")
        st.info(f"**Why the right answer works:** {q['why'][q['answer']]}")
    label = "See my results →" if pos + 1 == total else "Next question →"
    st.button(label, type="primary", key=f"quiz_next_{pos}", on_click=_next)


def _quiz_results() -> None:
    answers = st.session_state["quiz_answers"]
    total = len(QUESTIONS)
    score = sum(1 for i, q in enumerate(QUESTIONS) if answers.get(i) == q["answer"])
    pct = round(100 * score / total)
    if not is_complete(PAGE_ID):
        mark_complete(PAGE_ID)
        st.rerun()

    c1, c2 = st.columns(2)
    c1.metric("Final score", f"{score} / {total}")
    c2.metric("Percentage", f"{pct}%")
    st.progress(score / total)
    if pct >= 90:
        st.success("🎉 **Outstanding!** You have a strong grasp of embeddings and semantic search.")
    elif pct >= 70:
        st.success("✅ **Well done!** You understand the core ideas. Review the questions you missed below.")
    elif pct >= 50:
        st.warning("👍 **Good start.** Revisit Modules 2–4, then retake the quiz.")
    else:
        st.warning("📚 **Keep going.** Re-read the key takeaways in each module and try the playground again, then retake the quiz.")
    st.markdown("**Module 6 is marked complete.** Try the *Final challenge* tab next.")

    with st.expander("Review your answers"):
        for i, q in enumerate(QUESTIONS):
            ok = answers.get(i) == q["answer"]
            st.markdown(f"{'✅' if ok else '❌'} **{i + 1}. {q['q']}**")
            st.caption(f"Your answer: {q['options'][answers[i]]}" + ("" if ok else f"  ·  Correct: {q['options'][q['answer']]}"))
    st.button("Retake quiz", key="quiz_retake", on_click=_reset_quiz)


# ----------------------------------------------------------------------------- challenge
def _challenge() -> None:
    st.markdown(
        "### Final challenge\n"
        "> **Create a semantic search engine for a university knowledge base.** Add at least five documents, test "
        "three different queries, compare keyword search with semantic search, and explain the results."
    )
    st.markdown("Use the tool you built in **Module 5** (tab ①, then tab ②), then record your findings here.")

    docs = [d for d in st.session_state.get("kb_docs", []) if len(str(d.get("text", "")).strip()) >= 10]
    mine = sum(1 for d in docs if d.get("mine"))
    log = st.session_state.get("kb_query_log", [])
    distinct = {e["query"].strip().lower() for e in log}
    compared = any(e["compared"] for e in log)
    built = bool(st.session_state.get("kb_index"))

    section("Automatic progress check (from your Module 5 activity)", "📡")
    rows = [
        (len(docs) >= 5, f"At least 5 documents in the knowledge base: **{len(docs)}** ({mine} written by you)"),
        (built, "Embeddings generated (index built)"),
        (len(distinct) >= 3, f"At least 3 different queries tested: **{len(distinct)}**"),
        (compared, "Keyword search compared with semantic search at least once"),
    ]
    for ok, text in rows:
        st.markdown(f"{'✅' if ok else '⬜'} {text}")
    if not st.session_state.get("kb_docs"):
        st.caption("Nothing detected yet. Open Module 5 and run your searches first.")

    section("Your observations", "✍️")
    notes = {}
    for n in (1, 2, 3):
        notes[f"q{n}"] = st.text_input(f"Query {n} (the question you tested)", key=f"ch_q{n}", max_chars=200)
        notes[f"o{n}"] = st.text_area(
            f"What did keyword search and semantic search each retrieve for query {n}, and why?",
            key=f"ch_o{n}", height=90, max_chars=1200,
        )
    notes["c"] = st.text_area(
        "Overall conclusion: when did semantic search help? Where did it struggle? What are its limits?",
        key="ch_conclusion", height=120, max_chars=1500,
    )

    section("Model checklist: evaluate your own work", "🧾")
    ticked = 0
    for i, item in enumerate(CHECKLIST):
        if st.checkbox(item, key=f"ch_check_{i}"):
            ticked += 1
    st.progress(ticked / len(CHECKLIST), text=f"{ticked} of {len(CHECKLIST)} checklist items met")

    with st.expander("See a model answer for comparison"):
        st.markdown(
            """
**Query:** *“Can I sit for exams if I missed too many classes?”*

**Keyword search:** returned **no results** with the sample documents. After removing filler words the keywords are
*sit, exams, missed, many, classes*, and none of those exact words appears in any document (the policy says
*examinations* and *attendance*). The document that answers the question was invisible.

**Semantic search:** the *Attendance Policy* should rank at or near the top. It shares almost no words with the question,
but its meaning (attendance rules for exam eligibility) is close to the question's meaning. Check the cosine score in your
own run and say whether it is high, moderate or low.

**Explanation:** keyword search is **lexical**, so it can only reward shared words. Embeddings are **semantic**: the
model places the question and the policy close together because similar ideas appear in similar contexts in its training data.

**Limits:** the score shows closeness, not correctness. The app retrieves documents; it doesn't verify that a policy is
current, and it always returns the closest documents even when none is relevant.

*(Your exact scores will differ. The reasoning is what matters.)*
"""
        )

    report = _build_report(notes, docs, distinct, ticked)
    st.download_button("⬇️ Download my observations (.md)", report, file_name="beyond-keywords-challenge.md",
                       mime="text/markdown", key="ch_download")


def _build_report(notes: dict, docs: list[dict], distinct: set, ticked: int) -> str:
    lines = [
        "# Beyond Keywords: Final Challenge Report",
        "",
        f"- Documents in knowledge base: {len(docs)}",
        f"- Distinct queries tested: {len(distinct)}",
        f"- Checklist items met: {ticked} / {len(CHECKLIST)}",
        "",
    ]
    for n in (1, 2, 3):
        lines += [f"## Query {n}: {notes.get(f'q{n}') or '(not filled in)'}", "", notes.get(f"o{n}") or "(no observation)", ""]
    lines += ["## Conclusion", "", notes.get("c") or "(no conclusion)", ""]
    return "\n".join(lines)
