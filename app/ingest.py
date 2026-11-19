"""PDF ingestion: extract -> clean -> chunk -> embed -> store in Chroma, with a JSON registry of papers."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .config import get_settings

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
