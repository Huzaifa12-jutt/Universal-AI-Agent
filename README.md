# 🤖 AI Utility Agent

A production-ready, multi-tool AI agent built with **FastAPI**, **Google Gemini**, **LangChain/Groq (RAG)**, and **Streamlit**. It automatically routes your message to the right tool — calculator, live weather, web search, Wikipedia, PDF chat, or general Gemini chat — and streams back a clean, structured answer with a typing effect.

Built as a 2026 AI Engineering portfolio project. 100% free/freemium services, fully deployable, and beginner-friendly to read.

---

## ✨ Features

| Tool | What it does | Trigger examples |
|---|---|---|
| 🤖 AI Chat | General-purpose Q&A / conversation (Gemini) | anything else |
| 📄 PDF Chat (RAG) | Answers questions from PDFs you upload, with page citations | `summarize this document`, `what skills are mentioned in the pdf?` |
| 🧮 Calculator | Basic arithmetic (add/subtract/multiply/divide) | `10+20`, `calculate 50*23` |
| 🌤️ Weather | Live temperature, humidity, description via OpenWeatherMap | `weather in Lahore` |
| 🔎 Web Search | Real-time DuckDuckGo search, summarised by Gemini | `latest AI news` |
| 📖 Wikipedia | Concise Wikipedia summaries | `who is Alan Turing` |

Everything runs from **one Streamlit app**: a ChatGPT-style chat with live streaming responses, tool badges, copy-to-clipboard, source citations, chat history/download, a full PDF manager (drag & drop upload, delete, rebuild index), dark/light theme, and toast notifications for every action.

---

## 🏗️ Architecture

```
                         Streamlit UI (Streamlit Community Cloud)
                                       │  HTTP (streamed NDJSON)
                                       ▼
                         FastAPI Backend (Render)
                                       │
                                Agent (rule-based router)
                                       │
        ┌───────────┬────────────┬────┴────────┬─────────────┬─────────────┐
        ▼           ▼            ▼              ▼             ▼             ▼
   Calculator    Weather   DuckDuckGo Search  Wikipedia     PDF RAG      Gemini
                                                            (LangChain +   (fallback +
                                                          FAISS + FastEmbed  summarisation)
                                                             + Groq)
```

The `Agent` (`app/agent.py`) is a lightweight, keyword/regex-based router — no extra LLM call is spent just deciding *which* tool to use. The PDF tool only activates once you've actually uploaded a document; otherwise mentioning "pdf" just falls through to Gemini. `agent.stream()` powers `/chat/stream`, which streams tokens for Gemini/PDF answers and returns structured tools (calculator/weather/etc.) instantly as a single event.

**Why two `requirements.txt` files?** The backend (`requirements.txt`) carries the heavy RAG stack (LangChain, FAISS, FastEmbed) and runs on Render. The frontend (`frontend/requirements.txt`) is deliberately lightweight — just `streamlit`, `requests`, `python-dotenv` — because it only talks to the backend over HTTP. This is what keeps Streamlit Community Cloud deployments fast and free of dependency/Python-version conflicts.

---

## 📁 Folder Structure

```
ai-agent/
├── app/
│   ├── main.py              # FastAPI app, routes, streaming, CORS, error handling
│   ├── agent.py              # Tool router (classify / run / stream)
│   ├── config.py              # All settings/env vars (single source of truth)
│   ├── models.py              # Pydantic request/response schemas
│   ├── services/
│   │   └── llm.py              # Gemini API wrapper (ask_llm + ask_llm_stream)
│   ├── rag/                    # PDF RAG pipeline (reused, not rewritten)
│   │   ├── loader.py              # PDF → LangChain Documents
│   │   ├── chunker.py              # Recursive text splitting
│   │   ├── embedder.py              # FastEmbed local embeddings
│   │   ├── vector_store.py              # FAISS vector store
│   │   ├── retriever.py              # top-k retriever
│   │   ├── chatbot.py              # Groq (via langchain-openai) + prompt
│   │   ├── prompts.py              # RAG system prompt
│   │   └── manager.py              # Glues the above together + tracks uploaded files
│   └── tools/
│       ├── base.py              # Abstract BaseTool
│       ├── calculator.py
│       ├── weather.py
│       ├── search.py              # DuckDuckGo + Gemini summarisation
│       ├── wikipedia.py
│       ├── pdf_rag.py              # Thin wrapper around rag/manager.py
│       └── registry.py              # TOOLS dict
├── frontend/
│   ├── streamlit_app.py              # Full chat + PDF + settings UI
│   └── requirements.txt              # Lightweight deps for Streamlit Cloud
├── tests/
│   ├── test_tools.py
│   └── test_agent.py
├── .streamlit/config.toml              # Dark theme (startup default)
├── requirements.txt              # Backend deps (verified conflict-free)
├── .env.example
├── .gitignore
├── Dockerfile              # Backend container (for Render)
├── render.yaml              # Render Blueprint
└── run.py              # Runs backend + frontend together locally
```

---

## 🚀 Installation

```bash
git clone <your-repo-url>
cd ai-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | [Get a free key](https://aistudio.google.com/apikey) |
| `WEATHER_API_KEY` | ✅ | [Get a free key](https://openweathermap.org/api) |
| `GROQ_API_KEY` | ✅ (for PDF chat) | [Get a free key](https://console.groq.com/keys) |
| `GEMINI_MODEL` | – | Defaults to `gemini-flash-latest` |
| `GROQ_MODEL` | – | Defaults to `openai/gpt-oss-120b` |
| `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_TEMPERATURE` | – | Tune PDF retrieval/generation |
| `MAX_PDF_SIZE_MB`, `MAX_PDFS` | – | Upload limits (defaults: 20MB, 10 files) |
| `BACKEND_URL` | – | Used by the frontend; defaults to `http://127.0.0.1:8000` |
| `ALLOWED_ORIGINS` | – | CORS origins for the API, comma-separated (`*` for dev) |
| `LOG_LEVEL`, `DEBUG`, `REQUEST_TIMEOUT`, `SEARCH_MAX_RESULTS`, `WIKIPEDIA_SENTENCES` | – | See `app/config.py` |

**Never commit your real `.env` file** — it's already excluded via `.gitignore`.

> ℹ️ On its very first PDF upload, FastEmbed downloads a small (~130MB) embedding model from Hugging Face and caches it. This needs outbound internet access (fine on Render / your own machine) and takes a few extra seconds the very first time only.

---

## ▶️ Running Locally

Run backend + frontend together:

```bash
python run.py
```

Or run them separately (two terminals):

```bash
# Terminal 1 - backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 - frontend
streamlit run frontend/streamlit_app.py
```

- Backend docs (Swagger UI): http://127.0.0.1:8000/docs
- Frontend: http://localhost:8501

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All 14 tests pass (tool routing, PDF-gating logic, calculator/weather/search/wikipedia edge cases). Note: `pytest` and the RAG deps are backend-only — install `requirements.txt` (not `frontend/requirements.txt`) to run them.

---

## ☁️ Deployment

### Backend → Render

1. Push this repo to GitHub.
2. On [Render](https://render.com), click **New → Blueprint** and point it at your repo (it will read `render.yaml`), **or** manually create a **Web Service** using the included `Dockerfile`.
3. Add `GEMINI_API_KEY`, `WEATHER_API_KEY`, and `GROQ_API_KEY` as secrets in the Render dashboard.
4. Deploy. Render provides your backend URL, e.g. `https://ai-utility-agent-backend.onrender.com`.

### Frontend → Streamlit Community Cloud

1. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at **`frontend/streamlit_app.py`** as the entrypoint.
   - Streamlit Cloud auto-detects `frontend/requirements.txt` since it sits next to the entrypoint file — it will **not** pull in the heavy backend/RAG dependencies, so there's nothing to conflict.
2. In **Advanced settings**, pick **Python 3.11 or 3.12** explicitly before deploying (Streamlit Cloud's default version changes over time; the app doesn't need anything exotic, but pinning avoids surprises). This can only be set at deploy time — to change it later you must delete and redeploy the app.
3. In **App settings → Secrets**, add:
   ```toml
   BACKEND_URL = "https://ai-utility-agent-backend.onrender.com"
   ```
   (Streamlit Cloud exposes secrets as environment variables, which `streamlit_app.py` reads via `os.getenv`. You can also override this per-session from the app's own **Settings → Backend connection** field.)
4. Deploy. Your public chat UI is now live.

> ⚠️ Render's free tier spins down when idle — the first request after inactivity can take ~30-60s to wake up. The frontend shows a friendly "backend offline" status and clear timeout messages if this happens.

---

## 🧯 Error Handling

- Every tool (including PDF RAG) catches its own exceptions and returns `{"error": "..."}` instead of crashing.
- The FastAPI app has a global exception handler — no raw 500 stack traces ever reach the client.
- The Streamlit frontend checks backend health, shows a clear status pill, and surfaces readable timeout/connection-error messages instead of raw tracebacks.
- Invalid math (e.g. divide by zero), missing cities, ambiguous Wikipedia topics, empty queries, oversized/invalid PDF uploads, and "no PDF uploaded yet" are all handled explicitly.

## 📝 Logging

Structured logging is configured in `app/config.py::configure_logging()` and used throughout (`app/agent.py`, every tool, `app/rag/manager.py`, and `app/services/llm.py`). Every tool call, Gemini/Groq request, PDF index rebuild, and failure is logged with a timestamp and log level. Control verbosity via `LOG_LEVEL` in `.env`.

---

## 🔮 Future Improvements

- Persist conversation history + PDF index to a database/object storage (currently in-memory per process)
- Per-user PDF workspaces (currently one shared knowledge base per backend instance)
- Automatic tool-selection via Gemini/Groq function calling instead of keyword routing
- User authentication + per-user chat history
- Rate limiting on the API

## 📸 Screenshots

_Add screenshots of the chat UI here after your first deploy._

---

## 🛠️ Tech Stack

**Backend:** Python, FastAPI, Google Gemini (`google-genai`), LangChain (`langchain-community`, `langchain-openai`, `langchain-text-splitters`), FAISS, FastEmbed, Groq (OpenAI-compatible endpoint), DuckDuckGo Search (`ddgs`), Wikipedia API, OpenWeatherMap, Pydantic
**Frontend:** Streamlit
**Deployment:** Render (backend), Streamlit Community Cloud (frontend)

---

Made with ❤️ as part of an AI Engineering portfolio.
