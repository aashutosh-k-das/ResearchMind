# ResearchMind — Chat with your research papers

Upload a folder of research-paper PDFs, and ResearchMind chunks, embeds, and indexes them. You can then ask questions in plain language and get answers where **every claim is cited** back to the paper and page it came from.

Runs **fully locally** with Ollama (free, private), or on OpenAI by changing one environment variable.

**Stack:** FastAPI · LlamaIndex · ChromaDB · Ollama / OpenAI · vanilla JS UI (no build step)

---

## Features

- **Bulk ingestion:** drag-and-drop PDFs in the UI, or `python -m app.cli ingest <folder>` for 100+ papers.
- **Cited answers:** numbered `[n]` citations. Clicking one jumps to the source card, and **open ↗** opens the PDF at the cited page.
- **Streaming:** tokens stream over Server-Sent Events, and sources show up before the answer starts.
- **Scoped search:** tick papers in the sidebar to restrict retrieval to them.
- **Conversational follow-ups:** follow-up questions ("what are *its* limitations?") are rewritten into standalone search queries before retrieval.
- **Retrieval diversity:** a per-paper chunk cap stops one long paper from crowding out the others. If only one paper is relevant, its extra chunks fill the remaining slots.
- **Idempotent ingestion:** papers are keyed by content hash, so re-uploading the same PDF is a no-op, even under a different filename.
- **Background jobs:** uploads return `202` with a job ID you can poll for progress, so the API stays responsive during indexing.
- **Smart title extraction:** finds the largest-font text at the top of page 1, handles small caps and words hyphenated across lines, and ignores arXiv stamps and figure labels.
- **Provider-agnostic:** each embedding model gets its own Chroma collection, so switching between Ollama and OpenAI never mixes incompatible vectors.
- **Reasoning-model support:** chain-of-thought from models like `qwen3` and `deepseek-r1` is hidden from answers.

## Architecture

```
               ┌──────────── Ingestion (background worker) ─────────────┐
 PDF upload ──►│ pypdf extract → clean → title detect → SentenceSplitter │──► embeddings ──► ChromaDB
               │ (512 tok, 64 overlap) · sha256 content-hash dedupe       │                 (persistent)
               └─────────────────────────────────────────────────────────┘
                                                                                     │
 Question ──► condense w/ history ──► embed ──► top-k×3 vector search (+paper filter)│
                                              ──► per-paper diversity cap ──► top-k ─┘
                                              ──► numbered-source prompt ──► LLM (streamed)
                                              ──► answer + [n] citations + source cards
```

| File | Responsibility |
|---|---|
| [app/config.py](app/config.py) | All settings via env vars / `.env` (pydantic-settings) |
| [app/providers.py](app/providers.py) | LLM, embedding, and Chroma factories (Ollama ↔ OpenAI) |
| [app/ingest.py](app/ingest.py) | PDF extraction, chunking, indexing, registry, deletion |
| [app/rag.py](app/rag.py) | Retrieval, query condensing, grounded prompting, streaming |
| [app/main.py](app/main.py) | FastAPI routes, background ingestion jobs, SSE |
| [app/cli.py](app/cli.py) | `ingest` / `ask` / `list` from the terminal |
| [static/index.html](static/index.html) | Single-file chat UI |

## Quick start (local, free)

```bash
# 1. Models (one-time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Python env (Python 3.10–3.12; chromadb has no wheels for newer versions yet)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env

# 3. Run
./run.sh
# open http://localhost:8010
```

`run.sh` checks that the Python environment exists, that Ollama is running, and that the port is free before starting. To use a different port, pass it as an argument: `./run.sh 8020`.

Bulk-load a folder of papers:

```bash
.venv/bin/python -m app.cli ingest ~/Downloads/papers
.venv/bin/python -m app.cli ask "Which papers use contrastive learning, and how?"
```

### Use OpenAI instead

In `.env`:

```
PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Then re-ingest your papers. OpenAI vectors go into a separate collection.

### Docker

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Provider, models, paper/chunk counts |
| `GET` | `/api/documents` | List indexed papers |
| `POST` | `/api/documents/upload` | Multipart `files[]` → `202 {job_id}` |
| `POST` | `/api/documents/ingest-folder` | `{"path": "..."}` → `202 {job_id}` |
| `GET` | `/api/jobs/{job_id}` | Ingestion progress and per-file results |
| `DELETE` | `/api/documents/{doc_id}` | Remove a paper's vectors and file |
| `GET` | `/api/documents/{doc_id}/pdf` | Serve the PDF (append `#page=N`) |
| `POST` | `/api/query` | `{question, top_k?, doc_ids?, history?}` → `{answer, sources, cited}` |
| `POST` | `/api/query/stream` | Same body, SSE events: `sources` → `token`… → `done` |

Interactive docs: http://localhost:8010/docs

```bash
curl -s localhost:8010/api/query -H 'Content-Type: application/json' \
  -d '{"question": "How does LoRA compare to full fine-tuning on GPT-3?"}'
```

## Design notes

- **Why number the sources in the prompt instead of using a built-in citation engine?** It gives full control over the citation format. The server parses the `[n]` markers to report which sources were actually cited, and the UI dims the uncited ones.
- **Why `paper_id` and not `doc_id` in vector metadata?** LlamaIndex overwrites a node's `doc_id` metadata key with its own internal document UUID. Storing the content hash under `doc_id` silently broke filtering and deletion, so it lives under `paper_id`.
- **Why `nomic-embed-text` task prefixes?** The model was trained with `search_query:` and `search_document:` prefixes, and using them measurably improves retrieval.
- **Why a single ingestion worker?** It serializes writes to Chroma and the registry, so there are no write races, while query requests run concurrently.

## Known limitations & next steps

- **Small local models are the quality ceiling.** `llama3.2:3b` answers well when the right chunk is retrieved, but it can blur details or say "not stated" when a fact sits in a chunk that wasn't retrieved. Switching to `PROVIDER=openai` or a larger Ollama model (e.g. `qwen3:8b`) gives better answers.
- **Hybrid search (BM25 + vectors)** would help with exact terms such as model names, dataset names, and numbers.
- **Cross-encoder reranking** (e.g. `bge-reranker`) over the top-k×3 candidates.
- **Section-aware chunking** (Abstract / Method / Results), and dropping the References section.
- **OCR** for scanned PDFs, which are currently rejected with a clear error.
- **An evaluation harness** (a question → expected-paper set, measuring retrieval hit-rate and faithfulness).
