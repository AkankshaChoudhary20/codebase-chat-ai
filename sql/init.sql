-- codebase-chat-ai: schema initialization
-- Runs automatically on first container start via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repos (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    source_path_or_url  TEXT NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 384 dimensions matches sentence-transformers/all-MiniLM-L6-v2.
-- If you swap embedding models, update this dimension to match.
CREATE TABLE IF NOT EXISTS code_chunks (
    id            SERIAL PRIMARY KEY,
    repo_id       INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    symbol_name   TEXT,
    symbol_type   TEXT NOT NULL CHECK (symbol_type IN ('function', 'class', 'method', 'module')),
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    code_text     TEXT NOT NULL,
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS code_chunks_repo_id_idx ON code_chunks (repo_id);
CREATE INDEX IF NOT EXISTS code_chunks_file_path_idx ON code_chunks (repo_id, file_path);

-- HNSW index for fast approximate cosine-similarity search. Unlike IVFFlat,
-- HNSW doesn't need pre-existing data to build a useful index, which suits
-- a demo app where the table starts empty.
CREATE INDEX IF NOT EXISTS code_chunks_embedding_hnsw_idx
    ON code_chunks USING hnsw (embedding vector_cosine_ops);
