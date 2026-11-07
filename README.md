# 🧭 BEYOND KEYWORDS
### Understanding Embeddings & Semantic Search

> *"Teach machines to understand meaning, not just words."*

An interactive, self-paced learning lab (about **45 minutes**) that teaches undergraduates and beginners how
embeddings and semantic search work, by letting them experiment with **real model-generated embeddings** instead
of just reading about them. Built with Python and Streamlit. No paid API or API key is needed.

---

## Learning objectives

After completing the course, a learner can:

1. Explain why keyword (lexical) search fails when different words express the same idea.
2. Describe what an embedding is and generate real embeddings with Sentence Transformers.
3. Explain vector space and compute and interpret cosine similarity.
4. Compare keyword and semantic search on the same query and justify the differences.
5. Build a small semantic retrieval application (a university knowledge base).
6. State the limits: similarity is not factual correctness, and retrieval is not generation.

## Course outline

| # | Module | Time | What the learner does |
|---|--------|------|-----------------------|
| 1 | The Problem: Why Keyword Search Is Not Enough | 4 min | Runs a real keyword search and watches it miss the right documents |
| 2 | Understanding Embeddings | 8 min | Embeds two sentences, inspects the 384 numbers and the cosine score |
| 3 | Vector Space and Similarity | 7 min | Plays with angle vs. length; maps own sentences with PCA |
| 4 | Semantic Search Playground | 10 min | Compares keyword vs. semantic search; adds own documents |
| 5 | Build a Mini Semantic Search App | 10 min | Builds an index and searches a university knowledge base |
| 6 | Knowledge Check and Final Challenge | 6 min | 10-question quiz, then a practical challenge with a checklist |

Every module follows the same seven steps: **objective → simple explanation → analogy → practical example →
interactive activity → knowledge check → key takeaway.**

## Features

- Landing page, sidebar navigation, progress bar, per-module duration and completion status
- Real `all-MiniLM-L6-v2` embeddings; genuine cosine similarity; top-5 ranking
- Keyword vs. semantic search with clearly different score styles (**orange = keyword-match score**,
  **violet = cosine similarity**), so the two are never confused
- Interactive Plotly charts: embedding bars, an angle/length demo, a PCA map with labels
- Custom documents in the playground and in the mini project, with validation and cache invalidation
- Retrieval trace, live code walkthrough (reads the real source), and a RAG preview that builds a prompt
  **without calling any LLM**
- 10-question quiz with an explanation for every option, final score and percentage
- Final challenge with automatic progress detection, self-assessment checklist, model answer and a downloadable report
- Handles empty input, invalid documents, very long text and duplicate documents

## Technology stack

Python 3.10–3.12 · Streamlit · sentence-transformers (`all-MiniLM-L6-v2`) · NumPy · scikit-learn (PCA) · Plotly · Pandas

## Project structure

```
beyond-keywords/
├── app.py                      # entry point: page config, sidebar, routing
├── requirements.txt
├── README.md
├── .streamlit/config.toml      # dark theme
├── modules/
│   ├── home.py                 # landing page
│   ├── introduction.py         # Module 1
│   ├── embeddings.py           # Module 2
│   ├── vector_space.py         # Module 3
│   ├── semantic_playground.py  # Module 4
│   ├── mini_project.py         # Module 5
│   └── quiz.py                 # Module 6
├── utils/
│   ├── model.py                # cached model + cached embeddings
│   ├── similarity.py           # cosine similarity maths (pure NumPy)
│   ├── search.py               # keyword search, semantic search, RAG prompt preview
│   ├── data.py                 # sample documents and example queries
│   ├── progress.py             # page registry, navigation, completion tracking
│   └── ui.py                   # CSS, cards, score bars, knowledge checks, footer
├── tests/
│   ├── test_logic.py           # 24 unit tests, no model needed
│   └── test_model_integration.py   # tests with the real model
├── docs/
│   ├── SETUP_AND_DEPLOY.md     # VS Code (macOS), troubleshooting, testing checklist, GitHub + Streamlit Cloud
│   └── DESIGN_NOTES.md         # architecture and design decisions (interview prep)
└── assets/images/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The first search downloads the model (about 90 MB) and caches it; later runs are fast and work offline.

## Run the tests

```bash
python tests/test_logic.py                       # or: python -m pytest tests/test_logic.py -q
python -m pytest tests/test_model_integration.py -q -s   # needs the real model
```

## How the embedding model works

`all-MiniLM-L6-v2` is a small transformer neural network trained on over a billion sentence pairs so that
sentences with similar meaning produce similar vectors.

1. The text is split into **word-piece tokens**.
2. The network reads the tokens and outputs a vector for each one.
3. These are averaged ("mean pooling") into **one 384-number vector for the whole text**.

Embeddings are not tokens and not answers. They are learned numerical representations; no single dimension is
human-readable. The model reads at most 256 tokens, and longer text is cut off.

## How cosine similarity is calculated

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

- `A · B` is the dot product (multiply position by position, then add).
- `‖A‖ × ‖B‖` is the product of vector lengths. Dividing by it leaves **only direction**.
- Result: 1 = same direction, 0 = unrelated, −1 = opposite.

See `utils/similarity.py`. `cosine_similarity_matrix` scores a query against all documents with one matrix product.

## Application architecture

```
User query → Embedding model → Query vector → Cosine similarity vs. document vectors → Ranked documents → Retrieved results
                 ▲                                           ▲
          same model                       document embeddings computed once and cached
```

This is a **retrieval** system. It does **not** generate answers. In a RAG system, a language model would be added
after retrieval (see Module 5, tab ④).

## Real-world use cases

Customer-support FAQ matching · enterprise and document search · code search · recommendation systems ·
duplicate detection · RAG assistants over private knowledge bases.

## Limitations (stated honestly in the app)

- A high similarity score does not mean a document is factually correct or current.
- Semantic search always returns the *closest* documents, even when none are relevant. Low scores are a warning.
- The keyword baseline here is deliberately simple (exact words, no stemming or synonyms). Production keyword
  engines such as BM25 are stronger, but they are still lexical.
- Exact identifiers (order numbers, error codes) are often better served by keyword search.

## Future improvements

Chunking long documents · a vector database (FAISS/Chroma) · hybrid keyword + semantic ranking (BM25 + embeddings) ·
multilingual models · re-ranking with a cross-encoder · a RAG module with a real LLM · learner analytics and saved progress ·
accessibility audit and translations.

## Deployment

See `docs/SETUP_AND_DEPLOY.md` for the GitHub and Streamlit Community Cloud steps.
