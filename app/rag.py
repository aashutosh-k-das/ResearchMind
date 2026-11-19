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

import re
from dataclasses import asdict
from llama_index.core.llms import ChatMessage, MessageRole
from .providers import get_llm

SYSTEM_PROMPT = """You are ResearchMind, an assistant that answers questions about a library of research papers.

Rules:
- Answer ONLY using the numbered sources provided. Do not use outside knowledge.
- Cite every factual claim with the source number in square brackets, e.g. [1] or [2][4]. Put citations right after the claim they support.
- If the sources do not contain the answer, say so plainly: "The provided papers don't cover this." Do not guess.
- When papers disagree or report different numbers, point that out and cite each.
- Be concise and precise. Use short paragraphs or bullet points. Refer to papers by title when helpful."""

CONDENSE_PROMPT = """Given the conversation and a follow-up question, rewrite the follow-up as a standalone search query that contains all needed context (names of methods, papers, datasets). Output ONLY the rewritten question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""


def strip_thinking(text: str) -> str:
    return text.split("</think>", 1)[1].lstrip() if "</think>" in text else text


def condense_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up ("what about its limitations?") into a standalone retrieval query."""
    if not history:
        return question
    convo = "\n".join(f"{m['role'].title()}: {m['content'][:600]}" for m in history[-6:])
    out = strip_thinking(get_llm().complete(CONDENSE_PROMPT.format(history=convo, question=question)).text).strip()
    out = out.strip('"').splitlines()[0] if out else ""
    return out if 5 < len(out) < 500 else question


def _build_messages(question: str, sources: list[Source], history: list[dict]) -> list[ChatMessage]:
    context = "\n\n".join(
        f"[{src.id}] \"{src.title}\" (p. {src.page})\n{src.text}" for src in sources
    )
    msgs = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    for m in history[-6:]:
        role = MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT
        msgs.append(ChatMessage(role=role, content=m["content"]))
    msgs.append(
        ChatMessage(
            role=MessageRole.USER,
            content=f"Sources:\n\n{context}\n\n---\nQuestion: {question}\n\nAnswer with [n] citations:",
        )
    )
    return msgs


def cited_ids(answer: str, n_sources: int) -> list[int]:
    found = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n_sources)


NO_RESULTS = "I couldn't find anything relevant in the indexed papers. Try uploading papers first or rephrasing."


def answer(question: str, top_k: int | None = None, doc_ids: list[str] | None = None,
           history: list[dict] | None = None) -> dict:
    history = history or []
    query = condense_question(question, history)
    sources = retrieve(query, top_k, doc_ids)
    if not sources:
        return {"answer": NO_RESULTS, "sources": [], "cited": [], "search_query": query}
    text = strip_thinking(get_llm().chat(_build_messages(question, sources, history)).message.content or "")
    return {
        "answer": text,
        "sources": [asdict(s) for s in sources],
        "cited": cited_ids(text, len(sources)),
        "search_query": query,
    }
