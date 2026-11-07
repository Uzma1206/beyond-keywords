"""Shared look-and-feel: CSS, cards, callouts, score bars, knowledge checks, navigation."""
from __future__ import annotations

import html
import re

import streamlit as st

from utils.progress import (
    PAGE_IDS, go_to, is_complete, mark_complete, complete_and_go, page_by_id,
)

# Colour language (also used by Plotly charts)
NAVY = "#0B1020"
BLUE = "#4F8CFF"
VIOLET = "#8B5CF6"
AMBER = "#F5A524"
GREEN = "#34D399"
RED = "#F87171"
MUTED = "#9AA6C4"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(99,102,241,.20), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(59,130,246,.15), transparent 55%),
    #0B1020;
}
.block-container { max-width: 1080px; padding-top: 2rem; padding-bottom: 4rem; }
footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0E1530; border-right: 1px solid #1E2848; }
h1, h2, h3 { letter-spacing: -0.02em; }
code, pre { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

.bk-hero { padding: 2.6rem 2.2rem; border-radius: 22px; border: 1px solid #27325C;
  background: linear-gradient(135deg, rgba(79,140,255,.16), rgba(139,92,246,.20)); margin-bottom: 1.4rem; }
.bk-hero .eyebrow { color: #A5B4FC; font-size: .78rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; }
.bk-hero h1 { font-size: 3.1rem; line-height: 1.05; margin: .5rem 0 .2rem 0; font-weight: 800;
  background: linear-gradient(90deg, #E6EAF5, #A5B4FC 55%, #C4B5FD); -webkit-background-clip: text; background-clip: text; color: transparent; }
.bk-hero .sub { font-size: 1.15rem; color: #C7D0EA; margin: 0; }
.bk-hero .tag { margin-top: 1rem; font-size: 1.02rem; color: #9AA6C4; font-style: italic; }

.bk-stat { border: 1px solid #243055; background: #121A33; border-radius: 16px; padding: 1rem 1.1rem; text-align: left; }
.bk-stat .n { font-size: 1.7rem; font-weight: 700; color: #E6EAF5; }
.bk-stat .l { font-size: .85rem; color: #9AA6C4; }

.bk-card { border: 1px solid #243055; background: #121A33; border-radius: 16px; padding: 1.1rem 1.25rem; margin: .4rem 0 .9rem 0; }
.bk-card.done { border-color: rgba(52,211,153,.55); }
.bk-card .t { font-weight: 650; color: #E6EAF5; margin: .15rem 0 .25rem 0; }
.bk-card .d { color: #9AA6C4; font-size: .92rem; }
.bk-chip { display: inline-block; font-size: .72rem; font-weight: 600; padding: .16rem .6rem; border-radius: 999px;
  border: 1px solid #2C3A6B; color: #B8C4E8; background: rgba(79,140,255,.10); margin-right: .35rem; }
.bk-chip.ok { color: #34D399; border-color: rgba(52,211,153,.5); background: rgba(52,211,153,.10); }
.bk-chip.violet { color: #C4B5FD; border-color: rgba(139,92,246,.5); background: rgba(139,92,246,.12); }
.bk-chip.amber { color: #F5C26B; border-color: rgba(245,165,36,.5); background: rgba(245,165,36,.10); }

.bk-modhead { border: 1px solid #27325C; border-radius: 20px; padding: 1.5rem 1.6rem; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(79,140,255,.12), rgba(139,92,246,.14)); }
.bk-modhead .num { color: #A5B4FC; font-weight: 700; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; }
.bk-modhead h2 { margin: .3rem 0 .6rem 0; font-size: 1.85rem; font-weight: 750; }
.bk-modhead .obj { color: #C7D0EA; font-size: .98rem; }
.bk-modhead .obj b { color: #E6EAF5; }

.bk-label { font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #8B9AD0; margin: 1.7rem 0 .5rem 0; }

.bk-callout { border-radius: 14px; padding: .95rem 1.15rem; margin: .7rem 0 1rem 0; border: 1px solid #243055; background: #121A33; color: #D5DCF2; }
.bk-callout .ct { font-weight: 700; margin-bottom: .25rem; }
.bk-callout.analogy { border-color: rgba(139,92,246,.5); background: rgba(139,92,246,.10); }
.bk-callout.analogy .ct { color: #C4B5FD; }
.bk-callout.example { border-color: rgba(79,140,255,.5); background: rgba(79,140,255,.09); }
.bk-callout.example .ct { color: #93BBFF; }
.bk-callout.take { border-color: rgba(52,211,153,.55); background: rgba(52,211,153,.09); }
.bk-callout.take .ct { color: #34D399; }
.bk-callout.warn { border-color: rgba(245,165,36,.55); background: rgba(245,165,36,.09); }
.bk-callout.warn .ct { color: #F5C26B; }

.bk-result { border: 1px solid #243055; background: #121A33; border-radius: 14px; padding: .85rem 1.05rem; margin-bottom: .65rem; }
.bk-result .top { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; }
.bk-result .title { font-weight: 650; color: #E6EAF5; }
.bk-result .rank { color: #8B9AD0; font-weight: 700; margin-right: .5rem; }
.bk-result .score { font-family: 'JetBrains Mono', monospace; font-size: .9rem; white-space: nowrap; }
.bk-result .score.sem { color: #C4B5FD; }
.bk-result .score.kw { color: #F5C26B; }
.bk-result .body { color: #AEB9D8; font-size: .9rem; margin-top: .45rem; }
.bk-result .note { color: #8B9AD0; font-size: .8rem; margin-top: .35rem; }
.bk-result details summary { cursor: pointer; color: #8B9AD0; font-size: .8rem; margin-top: .3rem; }
.bk-bar { height: 8px; background: #1B2447; border-radius: 99px; overflow: hidden; margin-top: .5rem; }
.bk-bar > div { height: 100%; border-radius: 99px; }
.bk-bar .sem { background: linear-gradient(90deg, #4F8CFF, #8B5CF6); }
.bk-bar .kw { background: linear-gradient(90deg, #F5A524, #F59E0B); }
mark.bk-mark { background: rgba(245,165,36,.28); color: #FFD89A; padding: 0 .15rem; border-radius: 4px; }

.bk-flow { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; margin: .6rem 0 1rem 0; }
.bk-flow .node { border: 1px solid #2C3A6B; background: #141D3B; border-radius: 12px; padding: .5rem .8rem; font-size: .85rem; color: #DCE3F7; }
.bk-flow .node.hl { border-color: rgba(139,92,246,.7); background: rgba(139,92,246,.16); }
.bk-flow .node.off { opacity: .45; border-style: dashed; }
.bk-flow .arrow { color: #6C7AB0; }

.bk-opt { border: 1px solid #243055; border-radius: 10px; padding: .5rem .8rem; margin-bottom: .4rem; color: #D5DCF2; background: #121A33; }
.bk-opt.right { border-color: rgba(52,211,153,.7); background: rgba(52,211,153,.10); }
.bk-opt.wrong { border-color: rgba(248,113,113,.7); background: rgba(248,113,113,.10); }

@media (max-width: 760px) {
  .bk-hero { padding: 1.6rem 1.2rem; }
  .bk-hero h1 { font-size: 2.1rem; }
  .bk-modhead h2 { font-size: 1.4rem; }
  .bk-result .top { flex-direction: column; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def esc(text) -> str:
    """Escape user/document text for safe use inside our HTML blocks.

    '$' is escaped too, because Streamlit's markdown would otherwise try to
    treat text between two dollar signs as a LaTeX formula.
    """
    return html.escape(str(text), quote=False).replace("$", "&#36;")


def render_html(markup: str) -> None:
    """Render an HTML snippet. Lines are joined so markdown never sees indentation
    (4+ leading spaces would otherwise turn the block into a code block)."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def highlight_terms(text: str, terms: list[str]) -> str:
    """Escape text and wrap whole-word matches of `terms` in <mark>."""
    safe = esc(text)
    for term in sorted(set(terms), key=len, reverse=True):
        safe = re.sub(
            rf"(?<![A-Za-z0-9])({re.escape(term)})(?![A-Za-z0-9])",
            r'<mark class="bk-mark">\1</mark>',
            safe,
            flags=re.IGNORECASE,
        )
    return safe


def shorten(text: str, limit: int = 220) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def module_header(num: int, title: str, minutes: int, objective: str) -> None:
    render_html(
        f"""
        <div class="bk-modhead">
          <div class="num">Module {num} of 6 &nbsp;·&nbsp; ⏱ {minutes} min</div>
          <h2>{esc(title)}</h2>
          <div class="obj"><b>🎯 Learning objective:</b> {esc(objective)}</div>
        </div>
        """
    )


def section(label: str, icon: str = "") -> None:
    render_html(f'<div class="bk-label">{icon} {esc(label)}</div>')


def callout(kind: str, title: str, body_html: str) -> None:
    """kind: analogy | example | take | warn | info. body_html may contain simple tags."""
    render_html(f'<div class="bk-callout {kind}"><div class="ct">{esc(title)}</div>{body_html}</div>')


def flow(nodes: list[tuple[str, str]]) -> None:
    """Horizontal pipeline. nodes = [(label, style)], style in '', 'hl', 'off'."""
    parts = []
    for i, (label, style) in enumerate(nodes):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<span class="node {style}">{esc(label)}</span>')
    render_html(f'<div class="bk-flow">{"".join(parts)}</div>')


def bar_html(value: float, kind: str) -> str:
    pct = max(0.0, min(1.0, float(value))) * 100
    return f'<div class="bk-bar"><div class="{kind}" style="width:{pct:.0f}%"></div></div>'


def result_card(
    rank: int,
    title: str,
    category: str,
    text: str,
    score: float,
    kind: str,
    note: str = "",
    highlight: list[str] | None = None,
) -> None:
    """One search result. kind='sem' shows a COSINE score, kind='kw' a KEYWORD score."""
    if kind == "sem":
        score_label = f"cosine = {score:.3f}"
        badge = '<span class="bk-chip violet">Embedding similarity</span>'
    else:
        score_label = f"keywords = {score:.0%}"
        badge = '<span class="bk-chip amber">Keyword match</span>'
    body = highlight_terms(shorten(text, 230), highlight or [])
    more = ""
    if len(text) > 230:
        more = f"<details><summary>Read full document</summary><div class='body'>{highlight_terms(text, highlight or [])}</div></details>"
    note_html = f'<div class="note">{esc(note)}</div>' if note else ""
    render_html(
        f"""
        <div class="bk-result">
          <div class="top">
            <div><span class="rank">#{rank}</span><span class="title">{esc(title)}</span>
            <span class="bk-chip" style="margin-left:.5rem">{esc(category)}</span></div>
            <div class="score {kind}">{score_label}</div>
          </div>
          <div style="margin-top:.35rem">{badge}</div>
          {bar_html(score, kind)}
          <div class="body">{body}</div>
          {more}
          {note_html}
        </div>
        """
    )


def knowledge_check(key: str, question: str, options: list[str], correct: int, explanations: list[str]) -> None:
    """Multiple-choice check with immediate feedback and an explanation for EVERY option."""
    section("Knowledge check", "✅")
    state_key = f"kc_{key}"
    choice = st.radio(question, options, index=None, key=f"{state_key}_radio")
    if st.button("Check my answer", key=f"{state_key}_btn"):
        if choice is None:
            st.warning("Pick an answer first.")
        else:
            st.session_state[f"{state_key}_done"] = options.index(choice)
    picked = st.session_state.get(f"{state_key}_done")
    if picked is not None:
        if picked == correct:
            st.success(f"**Correct!** {explanations[picked]}")
        else:
            st.error(f"**Not quite.** {explanations[picked]}")
            st.info(f"**The right answer:** {options[correct]}. {explanations[correct]}")


def takeaway(text_html: str) -> None:
    section("Key takeaway", "💡")
    callout("take", "Remember this", text_html)


def module_footer(page_id: str) -> None:
    """Previous / mark complete / next navigation shown at the end of every module."""
    idx = PAGE_IDS.index(page_id)
    prev_id = PAGE_IDS[idx - 1]
    next_id = PAGE_IDS[idx + 1] if idx + 1 < len(PAGE_IDS) else None
    st.divider()
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c1:
        label = "← Home" if prev_id == "home" else f"← {page_by_id(prev_id)['short']}"
        st.button(label, key=f"prev_{page_id}", on_click=go_to, args=(prev_id,))
    with c2:
        if is_complete(page_id):
            st.success("Module complete ✔")
        else:
            st.button("✓ Mark module complete", key=f"done_{page_id}", on_click=mark_complete, args=(page_id,))
    with c3:
        if next_id:
            st.button(
                f"Next: {page_by_id(next_id)['short']} →",
                key=f"next_{page_id}",
                type="primary",
                on_click=complete_and_go,
                args=(page_id, next_id),
            )
        else:
            st.button("🏠 Back to home", key=f"home_{page_id}", type="primary", on_click=go_to, args=("home",))


def style_figure(fig, height: int = 420):
    """Apply the shared dark Plotly look."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,26,51,0.6)",
        font=dict(family="Inter, sans-serif", color="#D5DCF2"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig
