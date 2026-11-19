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
