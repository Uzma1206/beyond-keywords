"""Landing page."""
from __future__ import annotations

import streamlit as st

from utils.progress import MODULES, TOTAL_MINUTES, completed_count, go_to, is_complete
from utils.ui import callout, esc, render_html


def render() -> None:
    render_html(
        """
        <div class="bk-hero">
          <div class="eyebrow">Interactive learning lab · 45 minutes · self-paced</div>
          <h1>BEYOND KEYWORDS</h1>
          <p class="sub">Understanding Embeddings &amp; Semantic Search</p>
          <p class="tag">“Teach machines to understand meaning, not just words.”</p>
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, number, label in (
        (c1, "6", "short modules"),
        (c2, f"{TOTAL_MINUTES} min", "total, self-paced"),
        (c3, "Real", "model embeddings, no fake data"),
        (c4, "0", "API keys needed"),
    ):
        with col:
            render_html(f'<div class="bk-stat"><div class="n">{esc(number)}</div><div class="l">{esc(label)}</div></div>')

    st.markdown("")
    done = completed_count()
    if done == 0:
        st.button("Start Module 1 →", type="primary", key="home_start", on_click=go_to, args=("m1",))
    elif done < len(MODULES):
        nxt = next(m for m in MODULES if not is_complete(m["id"]))
        st.button(f"Continue with Module {nxt['num']} →", type="primary", key="home_continue", on_click=go_to, args=(nxt["id"],))
    else:
        st.success("🎓 You have completed every module. Revisit any module to experiment further.")

    st.markdown("### Your learning path")
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, m in zip(cols, MODULES[row_start:row_start + 3]):
            with col:
                complete = is_complete(m["id"])
                chip = '<span class="bk-chip ok">✔ Complete</span>' if complete else '<span class="bk-chip">Not started</span>'
                render_html(
                    f"""<div class="bk-card {'done' if complete else ''}">
                    <span class="bk-chip violet">Module {m['num']}</span><span class="bk-chip">⏱ {m['minutes']} min</span>{chip}
                    <div class="t">{m['icon']} {esc(m['title'])}</div><div class="d">{esc(m['blurb'])}</div></div>"""
                )
                st.button("Open", key=f"home_open_{m['id']}", on_click=go_to, args=(m["id"],))

    st.markdown("### How every module works")
    st.markdown(
        "Each module follows the same seven steps so you always know where you are: "
        "**objective → simple explanation → analogy → practical example → interactive activity → knowledge check → key takeaway.**"
    )
    callout(
        "info",
        "Before you start",
        "The first search or embedding downloads the small <b>all-MiniLM-L6-v2</b> model (about 90 MB) and may take up to a "
        "minute. After that it is cached and fast. Everything runs locally; nothing you type is sent to a paid service. "
        "This course teaches <b>semantic retrieval</b>. It does not generate answers with an LLM.",
    )
