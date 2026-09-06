"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(
        ...,
        min_length=1,
        description="A local filesystem path or a git URL (https://... or git@...).",
    )
    name: str | None = Field(
        default=None, description="Optional display name; defaults to the repo/dir name."
    )


class RepoOut(BaseModel):
    id: int
    name: str
    source_path_or_url: str
    ingested_at: datetime
    num_chunks: int


class IngestResponse(BaseModel):
    id: int
    name: str
    source: str
    files_scanned: int
    files_indexed: int
    chunks_created: int
    message: str = "Repo ingested successfully."


class CodeChunkOut(BaseModel):
    file_path: str
    symbol_name: str | None
    symbol_type: str
    start_line: int
    end_line: int
    location: str  # "path/to/file.py:10-42"
    code_text: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: list[CodeChunkOut]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A natural-language question about this codebase.")
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override the number of chunks to retrieve."
    )


class ChatResponse(BaseModel):
    answer: str
    cited_chunks: list[CodeChunkOut]


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str
    anthropic_configured: bool
