"""Postgres access layer (psycopg3 + pgvector).

A single module-level connection pool is created lazily on first use so
that importing this module (e.g. for tests, or for the chunker CLI) doesn't
require a live database connection.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from .config import get_settings

_pool: ConnectionPool | None = None


def _configure_connection(conn: psycopg.Connection) -> None:
    """Register the pgvector type adapter on every new pool connection."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            configure=_configure_connection,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def check_connection() -> bool:
    """Used by /health -- returns True if the DB is reachable."""
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------


def insert_repo(name: str, source_path_or_url: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO repos (name, source_path_or_url)
            VALUES (%s, %s)
            RETURNING id
            """,
            (name, source_path_or_url),
        ).fetchone()
        conn.commit()
        return row[0]


def get_repo(repo_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, name, source_path_or_url, ingested_at
            FROM repos
            WHERE id = %s
            """,
            (repo_id,),
        ).fetchone()
        if row is None:
            return None
        columns = ["id", "name", "source_path_or_url", "ingested_at"]
        return dict(zip(columns, row))


def list_repos() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.name, r.source_path_or_url, r.ingested_at,
                   COUNT(c.id) AS num_chunks
            FROM repos r
            LEFT JOIN code_chunks c ON c.repo_id = r.id
            GROUP BY r.id
            ORDER BY r.ingested_at DESC
            """
        ).fetchall()
        columns = ["id", "name", "source_path_or_url", "ingested_at", "num_chunks"]
        return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Code chunks
# ---------------------------------------------------------------------------


def insert_code_chunks(repo_id: int, chunks: list[dict[str, Any]]) -> int:
    """Bulk-insert chunks (each a dict from chunking.Chunk._asdict()-like shape)
    along with their pre-computed embeddings. Returns the number inserted.
    """
    if not chunks:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO code_chunks
                    (repo_id, file_path, symbol_name, symbol_type,
                     start_line, end_line, code_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        repo_id,
                        c["file_path"],
                        c["symbol_name"],
                        c["symbol_type"],
                        c["start_line"],
                        c["end_line"],
                        c["code_text"],
                        c["embedding"],
                    )
                    for c in chunks
                ],
            )
        conn.commit()
    return len(chunks)


def search_similar_chunks(
    repo_id: int, query_embedding: list[float], top_k: int = 8
) -> list[dict[str, Any]]:
    """Cosine-similarity nearest-neighbor search over a single repo's chunks.

    pgvector's `<=>` operator returns *cosine distance* (0 = identical,
    2 = opposite). We convert to a similarity score (1 - distance) for
    display purposes.

    Note: the explicit `::vector` cast on the parameter is required. Without
    it, Postgres resolves the `<=>` operator against the parameter's default
    inferred type (a plain float array), which has no matching operator --
    pgvector only provides an *assignment* cast from float arrays to
    `vector` (usable on INSERT/UPDATE), not one usable in operator
    resolution.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.file_path,
                c.symbol_name,
                c.symbol_type,
                c.start_line,
                c.end_line,
                c.code_text,
                1 - (c.embedding <=> %(query)s::vector) AS similarity
            FROM code_chunks c
            WHERE c.repo_id = %(repo_id)s
            ORDER BY c.embedding <=> %(query)s::vector
            LIMIT %(top_k)s
            """,
            {"query": query_embedding, "repo_id": repo_id, "top_k": top_k},
        ).fetchall()
        columns = [
            "id",
            "file_path",
            "symbol_name",
            "symbol_type",
            "start_line",
            "end_line",
            "code_text",
            "similarity",
        ]
        return [dict(zip(columns, row)) for row in rows]
