"""PDF ingestion: extract -> clean -> chunk -> embed -> store in Chroma, with a JSON registry of papers."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

from .config import get_settings
from .providers import get_chroma_collection, get_index

log = logging.getLogger(__name__)
_registry_lock = threading.Lock()


class IngestError(Exception):
    pass


# ---------------------------------------------------------------- registry
def load_registry() -> dict[str, dict]:
    path = get_settings().registry_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("Registry corrupted; starting fresh")
        return {}


def _save_registry(reg: dict) -> None:
    path = get_settings().registry_path
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    tmp.replace(path)  # atomic


def list_documents() -> list[dict]:
    reg = load_registry()
    return sorted(({"doc_id": k, **v} for k, v in reg.items()), key=lambda d: d["title"].lower())


# ---------------------------------------------------------------- extraction
def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_font_size(page) -> str:
    """Papers set the title in the largest font on page 1. Collect that run of text."""
    segs: list[tuple[float, float, str]] = []
    top_half = float(page.mediabox.height) * 0.5

    def visitor(text, cm, tm, font_dict, font_size):
        if not text.strip() or re.match(r"\s*arXiv:", text):
            return
        y = cm[5] + tm[5] * cm[3]
        if y < top_half:  # titles live at the top; ignore big labels inside figures further down
            return
        size = font_size * math.hypot(tm[2], tm[3]) * math.hypot(cm[2], cm[3])
        segs.append((size, y, text))

    page.extract_text(visitor_text=visitor)
    if not segs:
        return ""
    biggest = max(s for s, _, _ in segs)
    start = next(i for i, (s, _, _) in enumerate(segs) if s >= biggest - 0.1)
    parts: list[str] = []
    prev_size = prev_y = None
    for size, y, text in segs[start:]:
        if size < biggest * 0.75:  # small caps are ~80% of cap height; body/author text is far smaller
            break
        if prev_size is not None and size < prev_size - 0.5:
            text = text.lstrip()  # small-caps tail after a big capital ("L" + " OW") -> "LOW"
        if prev_y is not None and y < prev_y - 1 and parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1]  # word hyphenated across a line break
            text = text.lstrip()
        parts.append(text)
        prev_size, prev_y = size, y
    title = re.sub(r"\s+", " ", "".join(parts)).strip()
    return title if 8 <= len(title) <= 250 else ""


def guess_title(reader: PdfReader, first_page: str, fallback: str) -> str:
    try:
        if t := _title_from_font_size(reader.pages[0]):
            return t
    except Exception:  # noqa: BLE001
        pass
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    meta_title = meta_title.strip()
    if 8 <= len(meta_title) <= 250 and not meta_title.lower().endswith((".pdf", ".dvi", ".tex")) \
            and "untitled" not in meta_title.lower() and "microsoft word" not in meta_title.lower():
        return meta_title
    # Heuristic: first reasonably long line of page 1 that isn't an arXiv stamp / header
    for line in first_page.splitlines()[:15]:
        line = line.strip()
        if 15 <= len(line) <= 200 and not re.match(r"^(arxiv|preprint|published|proceedings|\d)", line, re.I):
            return line
    return fallback


def extract_pdf(path: Path) -> tuple[str, list[tuple[int, str]]]:
    try:
        reader = PdfReader(str(path))
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"Could not open PDF: {e}") from e
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:  # noqa: BLE001
            raise IngestError("PDF is encrypted") from e
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = clean_text(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single bad page shouldn't sink the paper
            txt = ""
        if len(txt) > 30:
            pages.append((i, txt))
    if not pages:
        raise IngestError("No extractable text (scanned PDF? run OCR first)")
    title = guess_title(reader, pages[0][1], fallback=path.stem.replace("_", " "))
    return title, pages


# ---------------------------------------------------------------- ingestion
def ingest_pdf(path: Path, original_name: str | None = None) -> dict:
    """Ingest a single PDF. Idempotent: identical file content is skipped."""
    s = get_settings()
    data = path.read_bytes()
    doc_id = file_hash(data)
    name = original_name or path.name

    reg = load_registry()
    if doc_id in reg:
        return {"doc_id": doc_id, "status": "skipped", "reason": "already ingested", **reg[doc_id]}

    t0 = time.perf_counter()
    title, pages = extract_pdf(path)

    documents = []
    for page_no, text in pages:
        doc = Document(
            text=text,
            # NB: not "doc_id" -- LlamaIndex overwrites that key with its internal ref-doc UUID.
            metadata={"paper_id": doc_id, "file_name": name, "title": title, "page": page_no},
            # Title helps the embedding place chunks in context; ids/page numbers are noise.
            excluded_embed_metadata_keys=["paper_id", "file_name", "page"],
            excluded_llm_metadata_keys=["paper_id"],
        )
        documents.append(doc)

    splitter = SentenceSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    for n in nodes:
        n.excluded_embed_metadata_keys = ["paper_id", "file_name", "page"]

    get_index().insert_nodes(nodes)

    entry = {
        "file_name": name,
        "title": title,
        "pages": len(pages),
        "chunks": len(nodes),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _registry_lock:
        reg = load_registry()
        reg[doc_id] = entry
        _save_registry(reg)
    log.info("Ingested %s (%d chunks) in %.1fs", name, len(nodes), time.perf_counter() - t0)
    return {"doc_id": doc_id, "status": "ingested", "seconds": round(time.perf_counter() - t0, 1), **entry}


def ingest_folder(folder: Path, progress=None) -> list[dict]:
    pdfs = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".pdf")
    results = []
    for i, p in enumerate(pdfs, start=1):
        try:
            results.append(ingest_pdf(p))
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to ingest %s", p)
            results.append({"file_name": p.name, "status": "error", "error": str(e)})
        if progress:
            progress(i, len(pdfs), results[-1])
    return results


def delete_document(doc_id: str) -> bool:
    with _registry_lock:
        reg = load_registry()
        if doc_id not in reg:
            return False
        get_chroma_collection().delete(where={"paper_id": doc_id})
        entry = reg.pop(doc_id)
        _save_registry(reg)
    stored = get_settings().papers_dir / entry["file_name"]
    if stored.exists() and file_hash(stored.read_bytes()) == doc_id:
        stored.unlink()
    return True
