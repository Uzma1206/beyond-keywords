# Design Notes: architecture and decisions (interview preparation)

## Architecture

```
app.py ──► sidebar (progress, nav radio) ──► ROUTES[page].render()
              │
              ├── modules/*.py     one file per lesson, each exposes render()
              └── utils/
                    model.py       st.cache_resource model + st.cache_data embeddings
                    similarity.py  pure NumPy cosine maths      ┐ no Streamlit imports,
                    search.py      keyword + semantic ranking   ┘ so they are unit-tested
                    ui.py          one visual language: CSS, cards, bars, knowledge checks
                    progress.py    page registry + completion state (st.session_state)
                    data.py        sample documents / example queries
```

## Key decisions and how to defend them

**1. Pure logic is separate from the UI.** `similarity.py` and `search.py` import no Streamlit, so 24 unit tests run
in milliseconds without downloading a model. For an educational product, correctness of the maths is the most
important thing to prove.

**2. Two kinds of caching, on purpose.** The model is a large shared object, so `st.cache_resource` (created once,
never copied). Embeddings are data, so `st.cache_data`, keyed by the **tuple of texts**. When the document
collection changes, the key changes, so the cache invalidates itself. Nothing to forget, no stale vectors.

**3. Same model for queries and documents.** Vectors from different models live in different spaces. The code
checks dimensions and raises a clear error on a mismatch.

**4. Cosine similarity, implemented by hand, not hidden.** `cosine_similarity_matrix` computes A·B / (‖A‖‖B‖) for
all documents with one matrix product, handles zero vectors (returns 0 instead of crashing), and clips floating-point
drift into [−1, 1]. It is tested against the scalar version.

**5. Keyword and semantic scores look different on purpose.** Keyword = orange bar, "keywords = 67%". Semantic =
violet bar, "cosine = 0.512". Different colour, label and chip, so learners never mistake one for the other. This
was an explicit requirement and a common beginner confusion.

**6. The keyword baseline is intentionally simple** (exact words, stopwords removed, no stemming). The app says so.
That keeps the lexical-vs-semantic lesson honest: it shows the property of *lexical* matching, not that BM25 is bad.

**7. Honest about limitations.** The app repeatedly states that similarity ≠ factual correctness, that semantic search
always returns the closest documents (so low scores are a warning), that embeddings approximate meaning, and that
Module 5 is retrieval, not generation. The RAG tab only builds a prompt string; no LLM is called.

**8. PCA is explained as a projection.** Module 3 reports the share of variation kept by 2 axes and computes how
often the map's nearest neighbour is the *true* (384-D cosine) nearest neighbour, so learners see the distortion
rather than being told about it.

**9. Instructional design.** Every module uses the same seven-step pattern, ordered problem → concept → visual →
experiment → application. Time budgets sum to exactly 45 minutes (4+8+7+10+10+6). Each module has a knowledge check
with feedback on *every* option (so wrong answers teach, too). Module 6 detects what the learner actually did in
Module 5 (documents, queries, comparison), so the challenge is tied to real activity, not just self-report.

**10. Robustness.** Empty or whitespace input, filler-only queries, invalid/too-short/symbol-only/duplicate documents,
2,000-character cap, long text (token limit warning), dimension mismatch, zero vectors, and an empty collection are
all handled and tested.

**11. Code walkthrough shows real code.** Module 5 uses `inspect.getsource` so the displayed code can never drift from
what runs.

## Likely interview questions

- *Why 384 dimensions?* That is the output size of all-MiniLM-L6-v2 (a design trade-off between quality and speed).
- *Why cosine and not Euclidean distance?* Direction carries topic information; length varies for unrelated reasons.
- *Why does semantic search return something for nonsense queries?* It ranks by closeness only. Add a minimum-similarity threshold (Module 5 has one).
- *How would this scale?* Replace the NumPy loop with a vector database / approximate nearest neighbour index (FAISS, Chroma, pgvector) and chunk long documents.
- *How would you improve retrieval quality?* Hybrid BM25 + embeddings, re-ranking with a cross-encoder, domain-tuned models.
- *Is this RAG?* No: it is the retrieval half. RAG adds chunking and an LLM that answers from retrieved text with citations.
- *How did you evaluate learning?* Per-module knowledge checks, a 10-question quiz and a performance-based challenge with a rubric.

## Known limits of this build

- Not yet tested in a live browser: development ran without Streamlit installed, so visual layout and the exact
  look of the CSS need a quick check on your machine.
- Progress is stored per browser session; it resets on refresh. Persisting it would need a database or login.
- Similarity thresholds in the interpretation bands are rules of thumb for this model, not universal values.
