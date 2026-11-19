"""LLM / embedding / vector store factories. Swap providers via the PROVIDER env var."""
from functools import lru_cache

import chromadb
from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import get_settings


@lru_cache
def get_llm():
    s = get_settings()
    if s.provider == "openai":
        from llama_index.llms.openai import OpenAI

        if not s.openai_api_key:
            raise RuntimeError("PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAI(model=s.openai_llm_model, api_key=s.openai_api_key, temperature=s.llm_temperature,
                      max_tokens=s.max_answer_tokens, timeout=s.request_timeout)

    from llama_index.llms.ollama import Ollama

    reasoning = any(h in s.ollama_llm_model.lower() for h in ("qwen3", "deepseek-r1", "qwq", "magistral", "gpt-oss"))
    return Ollama(
        model=s.ollama_llm_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        request_timeout=s.request_timeout,
        context_window=8192,
        thinking=False,
        additional_kwargs={
            # Reasoning models spend tokens "thinking" before answering, so give them more room.
            "num_predict": s.max_answer_tokens * (4 if reasoning else 1),
            "repeat_penalty": 1.1,
        },
    )


@lru_cache
def get_embed_model():
    s = get_settings()
    if s.provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=s.openai_embed_model, api_key=s.openai_api_key, embed_batch_size=64)

    from llama_index.embeddings.ollama import OllamaEmbedding

    kwargs = {}
    if "nomic" in s.ollama_embed_model:
        # nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
        kwargs = {"query_instruction": "search_query:", "text_instruction": "search_document:"}
    return OllamaEmbedding(
        model_name=s.ollama_embed_model, base_url=s.ollama_base_url, embed_batch_size=32, **kwargs
    )


@lru_cache
def get_chroma_collection():
    s = get_settings()
    client = chromadb.PersistentClient(path=str(s.chroma_dir))
    return client.get_or_create_collection(s.collection_name, metadata={"hnsw:space": "cosine"})


@lru_cache
def get_index() -> VectorStoreIndex:
    LISettings.llm = get_llm()
    LISettings.embed_model = get_embed_model()
    vector_store = ChromaVectorStore(chroma_collection=get_chroma_collection())
    storage = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage, embed_model=get_embed_model())
