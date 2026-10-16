# promptfoo RAG example

A minimal RAG app (TF-IDF retrieval + Google Gemini) evaluated with [promptfoo](https://www.promptfoo.dev).

## What is promptfoo?
An open-source CLI for testing LLM apps. You define test cases and assertions in YAML, run them, and compare results in a web UI. Useful for regression checks, prompt/model comparison, and red-teaming.

## Project layout
| File | Purpose |
|------|---------|
| `docs/*.txt` | Knowledge base (refunds, shipping, warranty) |
| `rag.py` | Retrieval (scikit-learn TF-IDF) + answer generation (Gemini) |
| `provider.py` | promptfoo Python provider that calls `rag.answer()` |
| `promptfooconfig.yaml` | Test cases and assertions (compares `gemini-2.5-flash` and `gemini-2.5-flash-lite`) |
| `prompt-eval.yaml` | Compares prompt wordings (`npx -y promptfoo@0.118.0 eval -c prompt-eval.yaml`) |
| `.env` | API keys (not committed) |

## Setup
```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```
Put `GOOGLE_API_KEY=...` in `.env`.

## Try the RAG app
```bat
python rag.py "What is the refund window?"
```

## Run the evaluation
```bat
set PROMPTFOO_PYTHON=%CD%\.venv\Scripts\python.exe
npx -y promptfoo@0.118.0 eval
```
(The pin to `0.118.0` is only needed on Node < 22.22.)

## Open the web viewer
```bat
npx -y promptfoo@0.118.0 view
```
Then open http://localhost:15500 (it usually opens automatically). Press `Ctrl+C` to stop.

## Assertions used
- `icontains` / `contains`: exact fact checks
- `llm-rubric`: LLM-as-judge grading (grader: `google:gemini-2.5-flash`)
