"""Module 3: Understanding Vector Space and Similarity (7 min)."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from utils.model import clean_text, embed_texts
from utils.similarity import cosine_similarity, interpret_similarity, pairwise_cosine_matrix
from utils.ui import (
    BLUE, VIOLET, callout, knowledge_check, module_footer, module_header, section,
    style_figure, takeaway,
)

PAGE_ID = "m3"
MAX_SENTENCES = 15
MIN_SENTENCES = 3
MAX_CHARS_PER_SENTENCE = 200

DEFAULT_SENTENCES = """I love playing cricket on weekends
The football team won the championship
She scored a goal in the final minute
My laptop is running very slowly
How can I speed up my computer?
The new smartphone has a faster processor
I need to revise for tomorrow's exam
Studying a little every day improves memory
Our professor explained the lecture clearly"""


def _pca_project(vectors: np.ndarray):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(vectors)
    return coords, pca.explained_variance_ratio_


def render() -> None:
    module_header(
        3,
        "Understanding Vector Space and Similarity",
        7,
        "Explain how cosine similarity compares embeddings and why a 2D map is only an approximation.",
    )

    section("1 · The simple explanation", "📖")
    st.markdown(
        """
Picture a huge map where every text is a **point** (or an arrow starting at the centre). That map is called
a **vector space**. Texts with similar meanings are placed near each other, and unrelated texts far apart.

To measure "nearness" for embeddings we usually use **cosine similarity**. It looks at the **angle** between
two arrows:
- arrows pointing the **same way** → similarity close to **1** (very similar)
- arrows at a **right angle** → close to **0** (unrelated)
- arrows pointing **opposite ways** → close to **−1** (opposite)
"""
    )
    st.latex(r"\text{similarity}(A,B)=\frac{A\cdot B}{\lVert A\rVert\times\lVert B\rVert}")
    st.markdown(
        """
**In plain language:**
- **A · B** (the *dot product*) multiplies the two vectors position by position and adds the results. It is large when
  the vectors agree.
- **‖A‖ × ‖B‖** is the product of the two vectors' *lengths*. Dividing by it removes the effect of length, so
  **only direction is compared**.
"""
    )
    with st.expander("Go deeper: why does direction matter more than length?"):
        st.markdown(
            """
Length can change for reasons unrelated to topic (for example, how long the text is). The *direction* is where
the topic information lives. Two arrows pointing the same way describe the same topic even if one is twice as long.
Many embedding models also return vectors that are already scaled to similar lengths, which makes cosine
similarity and a plain dot product agree.
"""
        )
    st.markdown("**How to read a score** (rough rules of thumb for this model, not universal laws):")
    st.table(
        {
            "Cosine score": ["≥ 0.65", "0.45 – 0.65", "0.25 – 0.45", "< 0.25"],
            "Typical meaning": ["Very similar", "Related", "Loosely related", "Mostly unrelated"],
        }
    )

    section("2 · A real-world analogy", "🧩")
    callout(
        "analogy",
        "Two people walking with compasses",
        "Two hikers set off from the same camp. Cosine similarity asks only <b>“are you heading in the same "
        "direction?”</b>, not <b>“how far have you walked?”</b>. One hiker can walk twice as far, and as long as "
        "they follow the same bearing, they are still going the same way.",
    )

    section("3 · A practical example: do the maths by hand", "🎓")
    a, b = np.array([1.0, 2.0]), np.array([2.0, 3.0])
    dot = float(np.dot(a, b))
    st.markdown(
        f"Let **A = [1, 2]** and **B = [2, 3]**.\n\n"
        f"- A · B = 1×2 + 2×3 = **{dot:.0f}**\n"
        f"- ‖A‖ = √(1² + 2²) = **{np.linalg.norm(a):.3f}**,  ‖B‖ = √(2² + 3²) = **{np.linalg.norm(b):.3f}**\n"
        f"- similarity = {dot:.0f} ÷ ({np.linalg.norm(a):.3f} × {np.linalg.norm(b):.3f}) = **{cosine_similarity(a, b):.3f}**"
    )
    st.caption("Real embeddings work exactly the same way, just with 384 numbers instead of 2.")

    section("4 · Try it yourself", "🕹️")
    st.markdown("#### Activity A: play with angle and length")
    _angle_demo()

    st.markdown("#### Activity B: map your own sentences")
    _pca_demo()

    knowledge_check(
        "m3a",
        "Vector B points in exactly the same direction as vector A but is three times longer. What is their cosine similarity?",
        ["About 0.33", "0", "1", "3"],
        correct=2,
        explanations=[
            "Length isn't used by cosine similarity, so the factor of 3 doesn't produce 0.33.",
            "0 means the vectors are at a right angle. These point the same way.",
            "Yes. Same direction means cosine similarity 1, whatever the lengths.",
            "Cosine similarity can never exceed 1.",
        ],
    )
    knowledge_check(
        "m3b",
        "Two sentences look far apart on the 2D PCA map. What can you safely conclude?",
        [
            "Their true similarity is definitely low",
            "The map is a squashed view of 384 dimensions, so check the cosine score for the real answer",
            "The sentences must be in different languages",
            "The model has made an error",
        ],
        correct=1,
        explanations=[
            "Squashing 384 dimensions into 2 can distort distances, so distance on the map isn't proof.",
            "Right. PCA keeps as much variation as it can in 2D, but some similarities get distorted.",
            "Language isn't implied by distance on the map.",
            "The map is a simplification, not evidence of a model error.",
        ],
    )

    takeaway(
        "Embeddings live in a <b>vector space</b>. <b>Cosine similarity</b> compares their <b>direction</b> (1 = same, "
        "0 = unrelated). A 2D plot is only a <b>shadow</b> of the real 384-dimensional picture."
    )
    module_footer(PAGE_ID)


def _angle_demo() -> None:
    c1, c2 = st.columns(2)
    with c1:
        angle = st.slider("Angle of vector B (degrees)", 0, 180, 40, key="vs_angle")
    with c2:
        length = st.slider("Length of vector B", 0.5, 3.0, 1.5, 0.1, key="vs_length")
    a = np.array([2.0, 0.0])
    theta = np.deg2rad(angle)
    b = length * np.array([np.cos(theta), np.sin(theta)])
    score = cosine_similarity(a, b)

    fig = go.Figure()
    for vec, name, colour in ((a, "A", BLUE), (b, "B", VIOLET)):
        fig.add_annotation(
            x=vec[0], y=vec[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=colour,
        )
        fig.add_trace(go.Scatter(x=[vec[0]], y=[vec[1]], mode="markers+text", text=[name],
                                 textposition="top center", marker=dict(size=9, color=colour),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(range=[-3.3, 3.3], zeroline=True)
    fig.update_yaxes(range=[-0.5, 3.3], scaleanchor="x", scaleratio=1, zeroline=True)
    fig.update_layout(title="Vector A (fixed) and vector B (you control it)")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(style_figure(fig, 360))
    with right:
        st.metric("Cosine similarity", f"{score:.3f}")
        st.metric("Dot product A · B", f"{float(np.dot(a, b)):.2f}")
        st.caption(f"‖A‖ = {np.linalg.norm(a):.2f}   ‖B‖ = {np.linalg.norm(b):.2f}")
        st.info("Drag **length** only: the dot product changes, but the cosine similarity doesn't. "
                "Drag **angle**: that's what moves the score.")


def _pca_demo() -> None:
    st.caption(
        f"Enter {MIN_SENTENCES}–{MAX_SENTENCES} sentences, one per line. The app embeds them with the real model and "
        "projects the 384 numbers down to 2 with PCA (Principal Component Analysis)."
    )
    st.session_state.setdefault("vs_text", DEFAULT_SENTENCES)
    with st.form("vs_form"):
        st.text_area("Sentences (one per line)", key="vs_text", height=210)
        submitted = st.form_submit_button("Map my sentences", type="primary")
    if submitted:
        st.session_state["vs_active"] = st.session_state["vs_text"]
    if "vs_active" not in st.session_state:
        st.session_state["vs_active"] = DEFAULT_SENTENCES

    raw_lines = [ln for ln in st.session_state["vs_active"].splitlines()]
    sentences, seen, trimmed = [], set(), False
    for ln in raw_lines:
        cleaned, cut = clean_text(ln, MAX_CHARS_PER_SENTENCE)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        sentences.append(cleaned)
        trimmed = trimmed or cut
    if len(sentences) > MAX_SENTENCES:
        st.warning(f"Using the first {MAX_SENTENCES} sentences only.")
        sentences = sentences[:MAX_SENTENCES]
    if trimmed:
        st.info(f"Long lines were shortened to {MAX_CHARS_PER_SENTENCE} characters.")
    if len(sentences) < MIN_SENTENCES:
        st.warning(f"Please enter at least {MIN_SENTENCES} different, non-empty sentences to draw a map.")
        return

    with st.spinner("Embedding sentences and projecting with PCA..."):
        vectors = embed_texts(tuple(sentences))
        coords, variance = _pca_project(vectors)
        sim = pairwise_cosine_matrix(vectors)

    labels = [textwrap.fill(s if len(s) <= 48 else s[:47] + "…", 24).replace("\n", "<br>") for s in sentences]
    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode="markers+text", text=labels,
            textposition="top center", textfont=dict(size=11),
            marker=dict(size=13, color=list(range(len(sentences))), colorscale=[[0, BLUE], [1, VIOLET]],
                        line=dict(width=1, color="#E6EAF5")),
            hovertext=[textwrap.fill(s, 50).replace("\n", "<br>") for s in sentences], hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Your sentences in 2D (PCA projection of real embeddings)",
        xaxis_title=f"Component 1 ({variance[0]:.0%} of variation)",
        yaxis_title=f"Component 2 ({variance[1]:.0%} of variation)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(style_figure(fig, 520))
    st.caption(
        f"The two axes together keep only **{variance.sum():.0%}** of the variation in the original 384 dimensions."
    )

    # Compare what the map shows with the true cosine scores
    n = len(sentences)
    sim_no_diag = sim.copy()
    np.fill_diagonal(sim_no_diag, -np.inf)
    dist2d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist2d, np.inf)
    agree = int(np.sum(np.argmax(sim_no_diag, axis=1) == np.argmin(dist2d, axis=1)))
    i, j = np.unravel_index(np.argmax(sim_no_diag), sim_no_diag.shape)
    # For the least-similar pair, the diagonal must not win, so mask it with +inf
    sim_for_min = np.where(np.isneginf(sim_no_diag), np.inf, sim_no_diag)
    p, q = np.unravel_index(np.argmin(sim_for_min), sim_for_min.shape)
    label_hi, _ = interpret_similarity(sim[i, j])

    c1, c2, c3 = st.columns(3)
    c1.metric("Most similar pair (true cosine)", f"{sim[i, j]:.2f}")
    c1.caption(f"“{sentences[i][:40]}” ↔ “{sentences[j][:40]}” ({label_hi})")
    c2.metric("Least similar pair (true cosine)", f"{sim[p, q]:.2f}")
    c2.caption(f"“{sentences[p][:40]}” ↔ “{sentences[q][:40]}”")
    c3.metric("Map's nearest neighbour is the true one", f"{agree} of {n}")
    c3.caption("Where this is below the total, the 2D map is misleading for those sentences.")

    callout(
        "warn",
        "The map is a projection, not the truth",
        "PCA squeezes 384 dimensions into 2, so some distances get distorted. Two points that look close may "
        "be less similar than they appear, and the reverse. The cosine scores (and the third box above) are the "
        "reliable numbers; the map is for building intuition.",
    )
    with st.expander("See the full cosine-similarity table (true scores in 384 dimensions)"):
        short = [s if len(s) <= 28 else s[:27] + "…" for s in sentences]
        st.dataframe(pd.DataFrame(np.round(sim, 2), index=short, columns=short))
