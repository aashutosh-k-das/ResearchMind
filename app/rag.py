"""Retrieval + grounded answer generation with numbered citations."""
from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .config import get_settings
from .providers import get_index, get_llm

log = logging.getLogger(__name__)

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
    # Over-fetch, then cap chunks per paper so one long paper can't crowd out the others.
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
    # If only one or two papers are relevant, backfill with their extra chunks rather than returning fewer.
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


def answer_stream(question: str, top_k: int | None = None, doc_ids: list[str] | None = None,
                  history: list[dict] | None = None, cancel: threading.Event | None = None) -> Iterator[dict]:
    """Yields events: {"type": "sources"}, many {"type": "token"}, then {"type": "done"}.

    Setting `cancel` (e.g. when the HTTP client disconnects) stops generation and closes the
    connection to the LLM, which makes Ollama abort instead of generating for nobody.
    """
    history = history or []
    query = condense_question(question, history)
    sources = retrieve(query, top_k, doc_ids)
    yield {"type": "sources", "sources": [asdict(s) for s in sources], "search_query": query}
    if not sources:
        yield {"type": "token", "text": NO_RESULTS}
        yield {"type": "done", "cited": []}
        return
    if cancel and cancel.is_set():
        return
    full = []
    think = ThinkFilter(_is_reasoning_model())
    stream = get_llm().stream_chat(_build_messages(question, sources, history))
    try:
        for chunk in stream:
            if cancel and cancel.is_set():
                log.info("Client disconnected; aborting generation")
                return
            if chunk.delta and (out := think.feed(chunk.delta)):
                full.append(out)
                yield {"type": "token", "text": out}
    finally:
        if cancel and cancel.is_set():
            log.info("Stream closed early after %d tokens; LLM generation stopped", len(full))
        stream.close()  # closes the HTTP stream to the LLM server -> generation stops there too
    if out := think.flush():
        full.append(out)
        yield {"type": "token", "text": out}
    yield {"type": "done", "cited": cited_ids("".join(full), len(sources))}


# ---------------------------------------------------------------- reasoning-model output
_REASONING_HINTS = ("qwen3", "deepseek-r1", "qwq", "magistral", "gpt-oss", "phi4-reasoning")


def _is_reasoning_model() -> bool:
    s = get_settings()
    return s.provider == "ollama" and any(h in s.ollama_llm_model.lower() for h in _REASONING_HINTS)


def strip_thinking(text: str) -> str:
    return text.split("</think>", 1)[1].lstrip() if "</think>" in text else text


class ThinkFilter:
    """Hides chain-of-thought from reasoning models that emit it inline, ending with </think>.

    For such models, output is held back until </think> appears; everything after it streams normally.
    If the stream ends without the tag, the whole buffer is released.
    """

    def __init__(self, active: bool):
        self.active, self.buf = active, ""

    def feed(self, delta: str) -> str:
        if not self.active:
            return delta
        self.buf += delta
        if "</think>" in self.buf:
            self.active = False
            out, self.buf = strip_thinking(self.buf), ""
            return out
        return ""

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out
