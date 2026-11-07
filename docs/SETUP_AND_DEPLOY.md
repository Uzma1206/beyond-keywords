# Setup, Testing and Deployment Guide

## 1. Create the project in VS Code on macOS

**Easiest:** unzip `beyond-keywords.zip` and open the folder (`File → Open Folder…`).

**Or build it by hand:**

```bash
mkdir beyond-keywords && cd beyond-keywords
mkdir -p .streamlit modules utils tests docs assets/images
touch app.py requirements.txt README.md .streamlit/config.toml
touch modules/__init__.py modules/home.py modules/introduction.py modules/embeddings.py \
      modules/vector_space.py modules/semantic_playground.py modules/mini_project.py modules/quiz.py
touch utils/__init__.py utils/model.py utils/similarity.py utils/search.py utils/data.py utils/progress.py utils/ui.py
code .
```

(`code .` needs VS Code's shell command: `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH".)
Then paste each file's content from the delivered project. Do not skip the two `__init__.py` files.

## 2. Virtual environment

Use **Python 3.10, 3.11 or 3.12** (PyTorch wheels lag behind the newest Python). Check with `python3 --version`.
If needed: `brew install python@3.12`.

```bash
cd beyond-keywords
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. In VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The install pulls in PyTorch, so it can take a few minutes and about 1–2 GB of disk.

## 4. Run

```bash
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. The first embedding action downloads the model (about 90 MB) and shows
"Loading the embedding model…". Stop the server with `Ctrl+C`.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: streamlit` | Activate the venv (`source .venv/bin/activate`) and reinstall |
| `ModuleNotFoundError: No module named 'modules'` / `'utils'` | Run from the project root (`streamlit run app.py` where `app.py` lives); make sure both `__init__.py` files exist |
| `ModuleNotFoundError: sentence_transformers` | Venv not active, or `pip install -r requirements.txt` did not finish |
| Pip fails building `torch` / "no matching distribution" | Use Python 3.10–3.12, then `pip install --upgrade pip` and retry |
| Model download hangs or fails (`ConnectionError`, SSL) | Check your network/VPN; retry. The download happens once and is cached in `~/.cache/huggingface` |
| First click is slow | Normal: model download plus load. Later runs are cached |
| App feels slow after editing | Streamlit reruns the script on every interaction; embeddings and the model are cached so it should stay fast |
| Port in use | `streamlit run app.py --server.port 8502` |
| Odd old state after code changes | In the app menu choose *Clear cache*, or press the sidebar *Reset my progress* button |
| macOS `urllib3`/OpenSSL warning | Harmless; upgrade with `pip install --upgrade urllib3` if it bothers you |

## 6. Testing checklist

Automated:

```bash
python tests/test_logic.py                              # 24 logic tests, no model needed
python -m pytest tests/test_model_integration.py -q -s  # real-model tests (downloads the model)
```

Manual walkthrough (run once before submitting):

- [ ] **Home:** hero, stats and six module cards appear; "Start Module 1" works.
- [ ] **Sidebar:** all pages open; progress bar and "minutes left" update when you complete modules; Reset works.
- [ ] **M1:** default query returns only the *battery* guide (keyword score orange). Empty query shows a message. "zebra" shows 0 results. The *Peek ahead* button shows violet cosine scores with a speed-up document above the battery guide.
- [ ] **M2:** each preset works; "Generate embeddings" shows 384 dimensions, token counts, 8 values, bar chart and a cosine score. Empty text shows a warning. Same-meaning preset scores clearly higher than "Unrelated topics".
- [ ] **M3:** sliders move the arrows; changing only *length* leaves cosine unchanged. "Map my sentences" shows a labelled PCA plot; the three topics roughly cluster. Fewer than 3 sentences shows a warning.
- [ ] **M4:** all three modes work; example buttons fill the query; orange vs. violet styling is clear; adding a custom doc then searching can rank it; invalid or duplicate docs are rejected; remove buttons work. A nonsense query shows the low-score warning.
- [ ] **M5:** Generate embeddings shows *N × 384*; adding/removing a doc marks the index stale until rebuilt; searching before building is blocked; threshold and top-k work; trace chart appears; code tab shows real source; RAG tab shows a prompt and says no LLM is called.
- [ ] **M6:** answer questions (immediate feedback, explanations for every option); final score and percentage appear; module is marked complete; retake works. In *Final challenge*, the automatic check reflects your Module 5 activity; checklist and download button work.

## 7. Upload to GitHub

```bash
cd beyond-keywords
git init
git add .
git commit -m "Beyond Keywords: interactive semantic search learning lab"
```

1. Create a new **public** repository on github.com (e.g. `beyond-keywords`). Do not add a README there.
2. Copy the two commands GitHub shows under *"…or push an existing repository"*:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/beyond-keywords.git
git push -u origin main
```

The included `.gitignore` keeps `.venv/` and `__pycache__/` out of the repository.

## 8. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub (authorise access to your repositories).
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose repository `<your-username>/beyond-keywords`, branch `main`, and **Main file path: `app.py`**.
4. (Optional) In *Advanced settings* pick Python 3.11 or 3.12.
5. Click **Deploy**. The first build installs PyTorch and Sentence Transformers (several minutes).

**Model download and memory considerations**

- The model (about 90 MB) downloads on the first embedding action after each cold start, so the very first click
  may take a minute. Streamlit's caching keeps it in memory afterwards.
- Community Cloud has limited memory (about 1 GB at the time of writing; check current limits). `all-MiniLM-L6-v2` is
  small, but the default PyTorch build is large. If the app crashes with an out-of-memory or install error,
  uncomment this line in `requirements.txt` to install the CPU-only PyTorch build, then push again:
  `--extra-index-url https://download.pytorch.org/whl/cpu`
- Apps go to sleep when idle; the model reloads on wake-up.
- Cached embeddings and the model are shared across visitors; each visitor's documents and progress are private
  to their own session.
