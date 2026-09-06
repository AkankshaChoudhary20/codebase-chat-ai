"""Local embedding generation using sentence-transformers.

Uses `all-MiniLM-L6-v2` (384-dim) by default -- small, fast, and runs
entirely on CPU with no external API calls or API keys required. This is
the same embedding approach as the sibling rag-docs-chat project; what
differs in this project is *what* gets embedded (code chunks split by
function/class boundary, not sliding-window text) -- see chunking.py.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers model.

    The model is downloaded from the Hugging Face hub on first use and
    cached locally afterwards (~90MB for all-MiniLM-L6-v2).
    """
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one 384-dim vector per input text."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
