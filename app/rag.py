"""Retrieval + grounded answer generation with numbered citations."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .config import get_settings
from .providers import get_index

log = logging.getLogger(__name__)


@dataclass
class Source:
    id: int
    doc_id: str
    title: str
    file_name: str
    page: int
    score: float
    text: str


def _filters(doc_ids: list[str] | None) -> MetadataFilters | None:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return MetadataFilters(filters=[MetadataFilter(key="paper_id", value=doc_ids[0], operator=FilterOperator.EQ)])
    return MetadataFilters(filters=[MetadataFilter(key="paper_id", value=doc_ids, operator=FilterOperator.IN)])


def retrieve(question: str, top_k: int | None = None, doc_ids: list[str] | None = None) -> list[Source]:
    s = get_settings()
    k = top_k or s.top_k
    retriever = get_index().as_retriever(similarity_top_k=k * 3, filters=_filters(doc_ids))
    hits = retriever.retrieve(question)

    per_doc: dict[str, int] = defaultdict(int)
    cap = s.max_chunks_per_doc
    chosen, overflow = [], []
    for h in hits:
        doc = h.node.metadata["paper_id"]
        if per_doc[doc] < cap:
            per_doc[doc] += 1
            chosen.append(h)
        else:
            overflow.append(h)
    chosen = sorted(chosen[:k] + overflow[: max(0, k - len(chosen))], key=lambda h: -(h.score or 0))
    chosen = chosen[:k]

    return [
        Source(
            id=i,
            doc_id=h.node.metadata["paper_id"],
            title=h.node.metadata.get("title", ""),
            file_name=h.node.metadata.get("file_name", ""),
            page=int(h.node.metadata.get("page", 0)),
            score=round(float(h.score or 0.0), 4),
            text=h.node.get_content(),
        )
        for i, h in enumerate(chosen, start=1)
    ]
