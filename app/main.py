"""ResearchMind API."""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import ingest
from .config import ROOT, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchmind")

app = FastAPI(title="ResearchMind", version="1.0.0", description="Chat with your research papers, with citations.")

# One worker => ingestion writes are serialized (Chroma + registry), while queries stay responsive.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------- models
class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None
    history: list[Turn] = []


class FolderRequest(BaseModel):
    path: str | None = None  # defaults to data/papers


# ---------------------------------------------------------------- jobs
def _new_job(kind: str, total: int) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "total": total, "done": 0,
                         "results": [], "created_at": time.time()}
    return job_id


def _update_job(job_id: str, **kw) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kw)


def _run_files(job_id: str, files: list[tuple[Path, str]]) -> None:
    _update_job(job_id, status="running")
    results = []
    for i, (path, name) in enumerate(files, start=1):
        try:
            results.append(ingest.ingest_pdf(path, original_name=name))
        except Exception as e:  # noqa: BLE001
            log.exception("Ingest failed: %s", name)
            results.append({"file_name": name, "status": "error", "error": str(e)})
            if path.exists() and path.parent == get_settings().papers_dir:
                path.unlink()  # don't keep files we couldn't index
        _update_job(job_id, done=i, results=list(results), current=name)
    _update_job(job_id, status="completed", current=None)


def _run_folder(job_id: str, folder: Path) -> None:
    _update_job(job_id, status="running")

    def progress(i, total, result):
        with _jobs_lock:
            _jobs[job_id]["results"].append(result)
            _jobs[job_id].update(done=i, total=total)

    try:
        ingest.ingest_folder(folder, progress=progress)
        _update_job(job_id, status="completed")
    except Exception as e:  # noqa: BLE001
        _update_job(job_id, status="failed", error=str(e))


def _safe_name(name: str) -> str:
    name = Path(name).name
    stem = re.sub(r"[^\w.\- ]+", "_", Path(name).stem).strip() or "paper"
    return f"{stem[:120]}.pdf"

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from .providers import get_chroma_collection


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "provider": s.provider,
        "llm": s.ollama_llm_model if s.provider == "ollama" else s.openai_llm_model,
        "embedding": s.ollama_embed_model if s.provider == "ollama" else s.openai_embed_model,
        "documents": len(ingest.load_registry()),
        "chunks": get_chroma_collection().count(),
    }


@app.get("/api/documents")
def documents():
    return ingest.list_documents()


@app.post("/api/documents/upload", status_code=202)
async def upload(files: list[UploadFile] = File(...)):
    s = get_settings()
    saved: list[tuple[Path, str]] = []
    rejected = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            rejected.append({"file_name": f.filename, "error": "not a PDF"})
            continue
        data = await f.read()
        if len(data) > s.max_upload_mb * 1024 * 1024:
            rejected.append({"file_name": f.filename, "error": f"larger than {s.max_upload_mb} MB"})
            continue
        if not data.startswith(b"%PDF"):
            rejected.append({"file_name": f.filename, "error": "file is not a valid PDF"})
            continue
        name = _safe_name(f.filename)
        dest = s.papers_dir / name
        if dest.exists() and ingest.file_hash(dest.read_bytes()) != ingest.file_hash(data):
            name = f"{ingest.file_hash(data)[:8]}_{name}"
            dest = s.papers_dir / name
        dest.write_bytes(data)
        saved.append((dest, name))
    if not saved:
        raise HTTPException(400, detail={"message": "No valid PDFs uploaded", "rejected": rejected})
    job_id = _new_job("upload", len(saved))
    _executor.submit(_run_files, job_id, saved)
    return {"job_id": job_id, "accepted": len(saved), "rejected": rejected}


@app.post("/api/documents/ingest-folder", status_code=202)
def ingest_folder(req: FolderRequest):
    folder = Path(req.path).expanduser().resolve() if req.path else get_settings().papers_dir
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {folder}")
    job_id = _new_job("folder", 0)
    _executor.submit(_run_folder, job_id, folder)
    return {"job_id": job_id, "folder": str(folder)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return dict(job)


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    if not ingest.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: str):
    entry = ingest.load_registry().get(doc_id)
    if not entry:
        raise HTTPException(404, "Document not found")
    path = get_settings().papers_dir / entry["file_name"]
    if not path.exists():
        raise HTTPException(404, "PDF file no longer on disk")
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{entry["file_name"]}"'})
