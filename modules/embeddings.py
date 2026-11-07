"""Module 2: Understanding Embeddings (8 min)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.model import (
    MODEL_NAME, clean_text, embed_texts, model_max_tokens, tokenize_for_model,
)
from utils.similarity import cosine_similarity, interpret_similarity
from utils.ui import (
    BLUE, VIOLET, bar_html, callout, esc, knowledge_check, module_footer, module_header,
    render_html, section, style_figure, takeaway,
)

PAGE_ID = "m2"

PRESETS = {
    "Same meaning, different words": (
        "I'm exhausted because I stayed up all night studying.",
        "I feel very tired after an all-night study session.",
    ),
    "Unrelated topics": (
        "The football match starts at seven o'clock.",
        "Photosynthesis converts sunlight into chemical energy.",
    ),
    "Same word, different meaning": (
        "The bank approved my education loan.",
        "We had a picnic on the bank of the river.",
    ),
}


def _set_pair(a: str, b: str) -> None:
    st.session_state["emb_a"] = a
    st.session_state["emb_b"] = b
    st.session_state.pop("emb_result", None)


def render() -> None:
    module_header(
        2,
        "Understanding Embeddings",
        8,
        "Describe how an embedding model turns text into numbers, and generate real embeddings yourself.",
    )

    section("1 · The simple explanation", "📖")
    st.markdown(
        """
An **embedding** is a list of numbers (a **vector**) that represents a piece of text. The list is always
the same length for a given model, whether the text is two words or two sentences.

**Why numbers?** Computers can't "read" meaning, but they are excellent at maths. If similar texts get
similar numbers, a computer can compare meaning using arithmetic.

**How is it made?** (1) The text is split into small pieces called *tokens*. (2) A neural network,
trained on millions of example sentences, reads the tokens. (3) It outputs one vector for the whole text.
In this course we use `all-MiniLM-L6-v2`, which outputs **384 numbers** per text.
"""
    )
    with st.expander("Go deeper: what do the 384 'dimensions' mean?"):
        st.markdown(
            """
A **dimension** is just one position in the list. A 2-number vector is a point on a flat map; a 3-number
vector is a point in space; 384 numbers is a point in a 384-dimensional space we can't picture.

Importantly, **no single dimension means something a person could name**, like "is about sports". The
model learned these dimensions on its own during training. Meaning lives in the *pattern across all 384
numbers*, not in any one of them.
"""
        )
    with st.expander("Go deeper: embeddings are not tokens, and not answers"):
        st.markdown(
            """
- **Tokens** are the small text pieces fed *into* the model (you can see them in the activity below).
- An **embedding** is what comes *out*: one fixed-length vector for the entire text.
- An embedding model does **not** write or answer anything. It only converts text to numbers. A chatbot
  (an LLM) is a different kind of model.
- Embeddings capture **statistical patterns learned from training data**, not human-level understanding.
"""
        )

    section("2 · A real-world analogy", "🧩")
    callout(
        "analogy",
        "Colours and map coordinates",
        "A screen describes <b>any colour</b> with just 3 numbers (red, green, blue). Two shades of blue have "
        "similar numbers; blue and orange don't. A GPS describes <b>any place</b> with 2 numbers, and nearby "
        "places have nearby numbers. An embedding does the same for <b>meaning</b>, using 384 numbers: "
        "texts with similar meaning get similar numbers.",
    )

    section("3 · A practical example", "🎓")
    st.markdown(
        "Two students write: *“I'm exhausted because I stayed up all night studying.”* and "
        "*“I feel very tired after an all-night study session.”* Almost no words match, yet any person sees "
        "they say the same thing. A good embedding model should place them **close together**. Let's check with the real model."
    )

    section("4 · Try it yourself", "🕹️")
    st.caption("Click a preset or write your own pair, then press **Generate embeddings**.")
    pc = st.columns(len(PRESETS))
    for col, (name, (a, b)) in zip(pc, PRESETS.items()):
        with col:
            st.button(name, key=f"emb_preset_{name}", on_click=_set_pair, args=(a, b))
    st.session_state.setdefault("emb_a", PRESETS["Same meaning, different words"][0])
    st.session_state.setdefault("emb_b", PRESETS["Same meaning, different words"][1])
    ca, cb = st.columns(2)
    with ca:
        st.text_area("Sentence A", key="emb_a", height=100)
    with cb:
        st.text_area("Sentence B", key="emb_b", height=100)

    if st.button("Generate embeddings", type="primary", key="emb_go"):
        a, a_cut = clean_text(st.session_state["emb_a"])
        b, b_cut = clean_text(st.session_state["emb_b"])
        if not a or not b:
            st.warning("Please fill in both sentences. Empty text has nothing to embed.")
            st.session_state.pop("emb_result", None)
        else:
            st.session_state["emb_result"] = {"a": a, "b": b, "cut": a_cut or b_cut}

    result = st.session_state.get("emb_result")
    if result:
        _show_result(result["a"], result["b"], result["cut"])

    knowledge_check(
        "m2a",
        "What does the embedding model output for one sentence?",
        [
            "A short summary of the sentence",
            "A fixed-length list of numbers (384 for this model)",
            "A list of the sentence's most important keywords",
            "An answer to the sentence",
        ],
        correct=1,
        explanations=[
            "Summaries are generated by language models. An embedding model only outputs numbers.",
            "Yes. Every text becomes the same-length vector, here 384 numbers.",
            "Keywords are words. Embeddings are numbers with no readable words in them.",
            "Embedding models don't answer anything. They only convert text into vectors.",
        ],
    )
    knowledge_check(
        "m2b",
        "Two sentences use completely different words but get a high similarity score. Why is that possible?",
        [
            "The model counts matching letters",
            "The model learned from huge amounts of text that these phrases are used in similar contexts",
            "The sentences must have the same number of tokens",
            "The model looked the sentences up in a dictionary",
        ],
        correct=1,
        explanations=[
            "Letter matching is what keyword search does. Embeddings capture learned patterns of usage.",
            "Correct. Training teaches the model that phrases used in similar situations should get similar vectors.",
            "Token count has nothing to do with meaning.",
            "There is no dictionary lookup. The vectors come from a trained neural network.",
        ],
    )

    takeaway(
        "An embedding turns text into a <b>fixed-length list of numbers</b> so similar meanings end up with "
        "similar numbers. The individual numbers are not human-readable. The <i>pattern</i> is what matters."
    )
    module_footer(PAGE_ID)


def _show_result(a: str, b: str, was_truncated: bool) -> None:
    with st.spinner("Running the embedding model..."):
        vectors = embed_texts((a, b))
        tokens_a, tokens_b = tokenize_for_model(a), tokenize_for_model(b)
        limit = model_max_tokens()
    vec_a, vec_b = vectors[0], vectors[1]
    score = cosine_similarity(vec_a, vec_b)
    label, meaning = interpret_similarity(score)

    if was_truncated:
        st.info("One of your texts was very long, so only its first 2,000 characters were used.")
    if max(len(tokens_a), len(tokens_b)) > limit:
        st.warning(
            f"The model only reads the first {limit} tokens of a text. Anything beyond that is ignored "
            "when the embedding is created."
        )

    st.markdown("##### What the model produced")
    m1, m2, m3 = st.columns(3)
    m1.metric("Embedding dimensions", f"{vec_a.shape[0]}")
    m2.metric("Tokens (A / B)", f"{len(tokens_a)} / {len(tokens_b)}")
    m3.metric("Cosine similarity", f"{score:.3f}")
    render_html(
        f"""<div class="bk-card"><span class="bk-chip violet">Cosine similarity (embedding model)</span>
        <b style="margin-left:.4rem">{esc(label)}</b> <span class="d">{esc(meaning)}</span>
        {bar_html(max(score, 0.0), "sem")}</div>"""
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Sentence A**")
        st.write(a)
        st.caption("Tokens the model saw: " + " · ".join(tokens_a[:30]) + (" …" if len(tokens_a) > 30 else ""))
    with right:
        st.markdown("**Sentence B**")
        st.write(b)
        st.caption("Tokens the model saw: " + " · ".join(tokens_b[:30]) + (" …" if len(tokens_b) > 30 else ""))

    st.markdown("**First 8 of the 384 numbers in each vector**")
    table = pd.DataFrame(
        {
            "Dimension": [f"#{i}" for i in range(8)],
            "Sentence A": [round(float(x), 4) for x in vec_a[:8]],
            "Sentence B": [round(float(x), 4) for x in vec_b[:8]],
        }
    )
    st.dataframe(table, hide_index=True)

    n_show = 48
    fig = go.Figure()
    fig.add_bar(x=list(range(n_show)), y=vec_a[:n_show], name="Sentence A", marker_color=BLUE)
    fig.add_bar(x=list(range(n_show)), y=vec_b[:n_show], name="Sentence B", marker_color=VIOLET)
    fig.update_layout(
        barmode="group", title=f"The first {n_show} dimensions of both embeddings",
        xaxis_title="Dimension index", yaxis_title="Value",
    )
    st.plotly_chart(style_figure(fig, 340))

    callout(
        "warn",
        "These numbers are not human-readable",
        "Don't try to read “dimension #3 = tiredness”. The values are coordinates the model learned; no single "
        "number has a name. What matters is that <b>similar texts produce similar overall patterns</b>, which is why "
        "the two bar groups tend to rise and fall together when the sentences are related. "
        "This is also a <i>learned approximation</i> of meaning, not human understanding.",
    )
    st.caption(f"Model: `{MODEL_NAME}`")
