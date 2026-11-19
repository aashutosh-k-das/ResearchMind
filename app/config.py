"""Central configuration, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Which backend to use for LLM + embeddings: "ollama" (local, free) or "openai".
    provider: Literal["ollama", "openai"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # OpenAI
    openai_api_key: str | None = None
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Storage
    papers_dir: Path = ROOT / "data" / "papers"
    chroma_dir: Path = ROOT / "storage" / "chroma"
    registry_path: Path = ROOT / "storage" / "registry.json"

    # Chunking / retrieval
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 6
    max_chunks_per_doc: int = 3  # diversity: avoid one paper flooding the context
    llm_temperature: float = 0.1
    request_timeout: float = 180.0
    # Hard cap on generated tokens. Small local models can fall into repetition loops;
    # without a cap one runaway answer would block Ollama for every later request.
    max_answer_tokens: int = 1024
    max_upload_mb: int = 50

    @property
    def collection_name(self) -> str:
        # Separate collections per embedding model: vectors from different models are not comparable.
        model = self.ollama_embed_model if self.provider == "ollama" else self.openai_embed_model
        safe = "".join(c if c.isalnum() else "_" for c in model)
        return f"papers_{self.provider}_{safe}"[:60]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.papers_dir.mkdir(parents=True, exist_ok=True)
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return s
