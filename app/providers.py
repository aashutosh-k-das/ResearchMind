"""LLM / embedding / vector store factories. Swap providers via the PROVIDER env var."""
from functools import lru_cache

import chromadb

from .config import get_settings


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
