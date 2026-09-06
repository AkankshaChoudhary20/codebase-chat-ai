"""Application configuration, loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central place for all configurable knobs.

    Every field can be overridden via an environment variable of the same
    name (case-insensitive) or via a `.env` file in the project root.
    """

    # --- Anthropic / Claude -------------------------------------------------
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = 2048

    # --- Postgres / pgvector -------------------------------------------------
    database_url: str = "postgresql://codeuser:codepassword@localhost:5433/codebasechat"

    # --- Embeddings ------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Retrieval ---------------------------------------------------------------
    top_k: int = 8

    # --- Ingestion -----------------------------------------------------------
    max_file_size_bytes: int = 1_000_000
    clone_workdir: str = "./data/clones"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
