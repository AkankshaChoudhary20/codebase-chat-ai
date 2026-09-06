"""Semantic search over a repo's indexed code chunks.

Shared by both `GET /repos/{id}/search` (returns ranked chunks directly, no
LLM call) and `POST /repos/{id}/chat` (feeds the same ranked chunks to
Claude as grounding context).
"""

from __future__ import annotations

from typing import Any

from . import db
from .embeddings import embed_query


def search_code(repo_id: int, query: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Embed `query` and return the top_k most similar code chunks for
    `repo_id`, ranked by cosine similarity (highest first).
    """
    query_embedding = embed_query(query)
    return db.search_similar_chunks(repo_id, query_embedding, top_k=top_k)


def format_location(chunk: dict[str, Any]) -> str:
    """Render a chunk's location as `path/to/file.py:10-42`, the label used
    both in the Claude prompt and in API responses.
    """
    return f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
