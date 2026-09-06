"""FastAPI app: ingest a codebase, search it semantically, or chat with it."""

from __future__ import annotations

from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, retrieval
from .config import get_settings
from .embeddings import get_embedder
from .ingestion import ingest_repo
from .llm import answer_question
from .schemas import (
    ChatRequest,
    ChatResponse,
    CodeChunkOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RepoOut,
    SearchResponse,
)

app = FastAPI(
    title="codebase-chat-ai",
    description="Chat with your codebase: code-aware RAG over any local repo or git URL.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _to_chunk_out(chunk: dict) -> CodeChunkOut:
    return CodeChunkOut(
        file_path=chunk["file_path"],
        symbol_name=chunk.get("symbol_name"),
        symbol_type=chunk["symbol_type"],
        start_line=chunk["start_line"],
        end_line=chunk["end_line"],
        location=retrieval.format_location(chunk),
        code_text=chunk["code_text"],
        similarity=chunk["similarity"],
    )


def _require_repo(repo_id: int) -> dict:
    repo = db.get_repo(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repo {repo_id} not found.")
    return repo


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        database="connected" if db.check_connection() else "unreachable",
        embedding_model=settings.embedding_model,
        anthropic_configured=bool(settings.anthropic_api_key),
    )


@app.post("/repos", response_model=IngestResponse)
def create_repo(payload: IngestRequest):
    """Ingest a repo: resolve the local path or git URL, walk its files,
    chunk each by function/class boundary, embed locally, and store in
    pgvector.
    """
    try:
        result = ingest_repo(payload.source, payload.name)
    except RuntimeError as exc:
        # Bad path, bad URL, clone failure, etc. -- caller's fault, not a
        # server error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        id=result.repo_id,
        name=result.name,
        source=result.source,
        files_scanned=result.files_scanned,
        files_indexed=result.files_indexed,
        chunks_created=result.chunks_created,
    )


@app.get("/repos", response_model=list[RepoOut])
def list_repos():
    return [RepoOut(**repo) for repo in db.list_repos()]


@app.get("/repos/{repo_id}/search", response_model=SearchResponse)
def search_repo(repo_id: int, q: str = Query(..., min_length=1), top_k: int = Query(default=None, ge=1, le=50)):
    """Pure semantic search -- no LLM call. Fast "find where X is
    implemented" lookup: embed the query, rank chunks by cosine similarity,
    return them with file:line citations.
    """
    _require_repo(repo_id)
    settings = get_settings()
    results = retrieval.search_code(repo_id, q, top_k=top_k or settings.top_k)
    return SearchResponse(query=q, results=[_to_chunk_out(r) for r in results])


@app.post("/repos/{repo_id}/chat", response_model=ChatResponse)
def chat_with_repo(repo_id: int, payload: ChatRequest):
    """Retrieval-augmented chat: embed the question, retrieve top-k chunks
    for this repo, ask Claude to answer citing file:line locations, and
    return both the answer and the chunks it was grounded on.
    """
    _require_repo(repo_id)
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    chunks = retrieval.search_code(repo_id, payload.question, top_k=top_k)

    try:
        answer = answer_question(payload.question, chunks)
    except RuntimeError as exc:
        # No ANTHROPIC_API_KEY configured at all.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except anthropic.AuthenticationError as exc:
        raise HTTPException(
            status_code=401, detail=f"Anthropic authentication failed: {exc.message}"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"Anthropic rate limit: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Anthropic API error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the Anthropic API.") from exc

    return ChatResponse(answer=answer, cited_chunks=[_to_chunk_out(c) for c in chunks])


@app.on_event("startup")
def warm_embedder() -> None:
    """Load the sentence-transformers model at startup rather than on the
    first request, so the first real ingest/search/chat isn't slow.
    """
    try:
        get_embedder()
    except Exception:
        # Don't crash the server if the model can't load (e.g. offline with
        # no cached model yet) -- endpoints that need it will raise a clear
        # error when actually called.
        pass
