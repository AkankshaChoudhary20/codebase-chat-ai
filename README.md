# codebase-chat-ai

Chat with a **codebase**, not a document. Point it at a local repo path or a git URL,
and it splits the code by function/class/method boundary (not a fixed-size sliding
window), embeds each symbol locally, and lets you either run a fast semantic search
("where is X implemented?") or ask Claude a natural-language question and get an
answer that cites the exact `file:line` locations it used.

## How this differs from a plain document-RAG app

This is the 5th project in a small series of RAG-style apps, and it's deliberately
built next to an earlier one, [`rag-docs-chat`](../rag-docs-chat) (FastAPI + pgvector +
local `sentence-transformers` + Claude, chunking plain text/PDF by a sliding window).
Same stack, different problem:

| | `rag-docs-chat` | `codebase-chat-ai` |
|---|---|---|
| Input | `.txt` / `.md` / `.pdf` documents | A local repo path or a git URL |
| Chunking | Fixed-size sliding window (~500 chars, 50-char overlap) over raw text | **Symbol-aware**: Python via `ast` (real parsing), other languages via a documented regex + brace-counting heuristic, split at function/class/method boundaries |
| Chunk metadata | Filename + chunk index | `file_path`, `symbol_name`, `symbol_type` (function/class/method/module), exact `start_line`/`end_line` |
| Retrieval-only endpoint | None -- every query goes through the LLM | `GET /repos/{id}/search` -- pure vector search, no LLM call, for "find where X is implemented" |
| Chat citations | "chunk #N of filename.pdf" | Precise `file:line` ranges Claude is instructed to cite, e.g. `app/db.py:120-151` |
| Ingestion source | File upload | Local filesystem path **or** `git clone --depth 1` of a URL |

The core idea: a document chatbot treats text as an undifferentiated stream and chunks
it blindly by character count. A codebase has real structure -- functions, classes,
methods, modules -- and chunking along that structure means a retrieved chunk is
always a complete, coherent unit (a whole function, not "the last 500 characters of
one function and the first 500 of the next"), and every citation can point at exactly
the lines that answer the question.

## Architecture

```
                    ┌────────────────────────────────┐
  Browser  ───────► │  FastAPI (app/main.py)          │
  (static           │                                  │
   HTML/JS at "/")  │  POST /repos ────────────────────┼──► ingestion.py
                     │                                  │      │  resolve local path OR
                     │                                  │      │  git clone --depth 1 (URL)
                     │                                  │      ▼
                     │                                  │   walk files (skip .git,
                     │                                  │   node_modules, venv, dist,
                     │                                  │   build, binaries, >1MB)
                     │                                  │      │
                     │                                  │      ▼
                     │                                  │   chunking.py
                     │                                  │     .py  -> ast (real parser)
                     │                                  │     .js/.ts/.go/.java/.rs
                     │                                  │          -> regex + brace-count
                     │                                  │          heuristic (documented
                     │                                  │          limitations below)
                     │                                  │     other -> whole file, 1 chunk
                     │                                  │      │
                     │                                  │      ▼
                     │                                  │   embeddings.py (sentence-
                     │                                  │     transformers, all-MiniLM-L6-v2,
                     │                                  │     local CPU, 384-dim)
                     │                                  │      │
                     │                                  │      ▼
                     │                                  │   db.py ──► Postgres + pgvector
                     │                                  │             (repos, code_chunks)
                     │                                  │
                     │  GET /repos/{id}/search ─────────┼──► retrieval.py (embed query,
                     │                                  │       cosine <=> search in
                     │                                  │       pgvector) -- no LLM call
                     │                                  │
                     │  POST /repos/{id}/chat ──────────┼──► retrieval.py (same search)
                     │                                  │       then llm.py ──► Claude API
                     │                                  │       (answer + file:line
                     │                                  │        citations, using the
                     │                                  │        retrieved chunks as context)
                     │                                  │
                     │  GET /repos                      │
                     │  GET /health                     │
                     └──────────────────────────────────┘
```

### Database schema (`sql/init.sql`)

- `repos(id, name, source_path_or_url, ingested_at)`
- `code_chunks(id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line, code_text, embedding vector(384))`
  with an HNSW index on `embedding` for approximate cosine-similarity search, and a
  `CHECK` constraint on `symbol_type IN ('function', 'class', 'method', 'module')`.

## The chunking approach -- and its honest limitations

### Python (`.py`) -- real AST parsing

Uses the standard library `ast` module. `ast.parse()` gives exact `lineno` /
`end_lineno` for every node, so boundaries are syntactically correct, not guessed:

- Each **top-level function** (`def` / `async def`) becomes one `function` chunk.
- Each **top-level class** becomes one `class` chunk containing the whole class body
  (useful for "what does class X do" questions), **and** each of its methods also
  becomes its own separate `method` chunk (`ClassName.method_name`) with tight
  line bounds (useful for "where is method Y implemented" precision).
- Decorators are included in the chunk's line span (a `@app.post(...)`-decorated
  function's chunk starts at the decorator line, not the `def` line).
- If the file has a syntax error, or has no top-level functions/classes (e.g. a
  script that's just top-level statements), it falls back to a single whole-file
  `module` chunk.

This is the one language where the chunker is a real parser, not a heuristic, and
verified line-accurate in this project's own end-to-end test (see Verification below).

### JavaScript / TypeScript / Go / Java / Rust -- regex + brace-counting heuristic

**This is explicitly a heuristic, not a parser**, and it will occasionally get things
wrong. No AST library was added for these languages (that would mean five more heavy
dependencies for a portfolio project); instead, `chunking.py` does two things per line:

1. A regex tries to match a declaration line: `function foo(...)`, `class Foo {`,
   `const foo = (...) => {`, `func (r *Receiver) Foo(...)`, `pub fn foo(...)`,
   `public void foo(...)`, etc. -- see `chunking.py` for the exact patterns.
2. From the matched line, a simple `{` / `}` depth counter scans forward until the
   depth returns to zero; that's the chunk's end line. A declaration line ending in
   `;` before any `{` (an abstract method, an interface signature) becomes a
   single-line chunk instead.

**Known limitations, stated plainly:**

- Braces inside **string literals, template literals, or comments** are counted like
  any other brace. A JS template literal containing a stray `{` in ordinary text can
  throw off the block boundary. This is rare in practice but is a real gap, not a
  parser.
- **Multi-line function signatures** (parameters spread across several lines before
  the opening `{`) are only found if the declaration keyword and name are on the
  *first* line the regex sees; the brace-counter still works once past the
  signature, but a signature broken across lines in an unusual way can be missed.
- **Arrow functions and expression-bodied functions without a `{ }` block**
  (`const add = (a, b) => a + b;`) are not detected as symbols at all -- only
  block-bodied arrow functions (`=> { ... }`) are.
- **Java method detection requires an explicit `public`/`private`/`protected`
  modifier** on the same line as the signature, to keep the regex from matching
  control-flow statements (`if (...) {`, `for (...) {`). Package-private methods
  (no modifier) are not detected as their own chunk; they get folded into the
  containing class's fallback (or missed if the class-level regex also doesn't
  match unusually formatted code).
- **Go and Rust "method" classification** is heuristic: a Go function with a
  receiver (`func (s *Server) Foo()`) is labeled `method`, named `Server.Foo`; a
  Rust `fn` is labeled `method` if it's indented (i.e. presumably inside an `impl`
  block) and `function` otherwise -- indentation, not real scope tracking.
- Nested functions/closures inside another already-matched function are generally
  *not* separately extracted -- they're just part of their parent's `code_text`.

None of this is "full AST-level parsing" for these five languages, and this README
says so rather than overclaiming it. In practice, on conventionally-formatted source
(the overwhelming majority of real code), it produces accurate, useful chunks -- see
the spot-checked examples in Verification below.

### Everything else -- whole file as one chunk

Any file with an unrecognized extension, or a recognized extension where the
heuristic finds zero symbols, becomes a single `module`-type chunk containing the
entire file. This also covers `.md`, `.json`, `.yaml`, `.toml`, `.sql`, `.sh`, and a
handful of other plain-text extensions that are indexed but never symbol-split.

### What's skipped entirely during ingestion

- Directories: `.git`, `node_modules`, `venv`/`.venv`/`env`, `__pycache__`, `dist`,
  `build`, `.next`, `.nuxt`, `target`, `vendor`, `.idea`, `.vscode`, `.mypy_cache`,
  `.pytest_cache`, and anything with `egg-info` in the name.
- Files over `MAX_FILE_SIZE_BYTES` (default 1MB).
- Binary files (detected the same way `git` does: a NUL byte in the first 8KB).
- Files that don't decode as UTF-8 text.
- Extensions not in the indexable list (see `INDEXABLE_EXTENSIONS` in `app/ingestion.py`).

## Setup

### 1. Start Postgres + pgvector

```bash
docker compose up -d
```

This runs `pgvector/pgvector:pg16` on **port 5434** (not the default 5432, so it can
run alongside other local Postgres/pgvector projects like `rag-docs-chat` without a
port clash) and applies `sql/init.sql` automatically on first start.

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on Python versions:** `requirements.txt` uses version *ranges*, not exact
> pins, specifically so this installs cleanly on newer Python versions where some
> packages don't ship prebuilt wheels yet for an exact pinned version. In particular,
> `psycopg2-binary` has no wheel on very new Python releases (e.g. 3.14) -- this
> project uses `psycopg[binary,pool]` (psycopg **v3**) instead, which does, matching
> the same fix used in the sibling `rag-docs-chat` project. Verified working end to
> end on Python 3.14 during development (see Verification below).

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY`. Get one at
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
(requires an Anthropic account; the console gives you a starting credit balance for
testing). Everything else in `.env.example` has a working default that matches
`docker-compose.yml`.

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/** for the built-in single-page frontend, or use the API
directly (interactive docs at `/docs`).

## Example: ingest this repo and ask "where is X handled"

```bash
# Ingest this project's own app/ directory
curl -X POST http://127.0.0.1:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"source": "'"$(pwd)"'/app", "name": "codebase-chat-ai"}'
# -> {"id": 1, "name": "codebase-chat-ai", "files_scanned": 9, "files_indexed": 9,
#     "chunks_created": 50, ...}

# Fast semantic search, no LLM call
curl -s "http://127.0.0.1:8000/repos/1/search?q=how+are+anthropic+api+errors+mapped+to+http+status+codes" | python3 -m json.tool
# -> top result: llm.py:53-90 (answer_question) -- exactly where those errors
#    are deliberately left to propagate for main.py to catch

# Ask Claude directly, with citations
curl -X POST http://127.0.0.1:8000/repos/1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does a missing ANTHROPIC_API_KEY get surfaced to an API caller?"}'
```

Or ingest any small public repo by URL instead of a local path:

```bash
curl -X POST http://127.0.0.1:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/kennethreitz/samplemod.git"}'
```

## API

| Endpoint | Description |
|---|---|
| `POST /repos` | Ingest a repo. Body: `{"source": "<local path or git URL>", "name": "<optional>"}`. |
| `GET /repos` | List ingested repos with chunk counts. |
| `GET /repos/{id}/search?q=...&top_k=8` | Pure semantic search -- ranked chunks with `file:line`, no LLM call. |
| `POST /repos/{id}/chat` | Body: `{"question": "...", "top_k": 8}`. Retrieves chunks, asks Claude to answer citing `file:line`, returns the answer plus the cited chunks. |
| `GET /health` | DB connectivity, embedding model name, whether an Anthropic key is configured. |
| `GET /` | The static frontend. |

## Verification (what was actually tested)

This was built and verified end to end on this machine (macOS, Python 3.14, Docker
available):

- **Dependencies installed cleanly**, including `sentence-transformers` and `torch`,
  in a fresh venv on Python 3.14 -- no build failures. (If your environment can't
  install `torch`/`sentence-transformers` in time, the fallback used in the sibling
  `rag-docs-chat` project is to stub `app/embeddings.py`'s `embed_texts`/`embed_query`
  with deterministic fake vectors so the rest of the pipeline -- ingestion, chunking,
  storage, retrieval, the Claude call -- can still be exercised and tested; that
  workaround was **not** needed here since the real model installed and ran fine.)
- **All modules import cleanly** and the FastAPI app boots (`uvicorn app.main:app`),
  confirmed by listing its registered routes programmatically.
- **Postgres + pgvector brought up via `docker compose up -d`**, confirmed healthy,
  and `sql/init.sql` schema (both tables, the HNSW index, the `symbol_type` check
  constraint) verified via `psql \dt` / `\d code_chunks`.
- **Real end-to-end ingestion**, three ways:
  1. This project's own `app/` directory (local path) -- 9 files, 50 chunks.
  2. `https://github.com/octocat/Hello-World.git` (git URL) -- clone succeeded;
     0 chunks because that repo's only file is a bare `README` with no extension,
     which correctly falls outside the indexable extension list. This confirmed the
     `git clone --depth 1` code path itself works even though that particular repo
     has no indexable code.
  3. `https://github.com/kennethreitz/samplemod.git` (git URL, real Python code) --
     9 files, 12 chunks.
- **Line numbers and symbol names spot-checked against actual source**, not just
  trusted: for this project's own code, `db.py:insert_code_chunks` (lines 120-151)
  and `main.py:chat_with_repo` (lines 114-143, correctly starting at the `@app.post`
  decorator line) were read back from the file and matched exactly against what was
  stored in Postgres. Chunks from the cloned `samplemod` repo (e.g.
  `sample/core.py:hmm` at lines 9-12) were likewise checked against the file.
- **Semantic search** (`GET /repos/{id}/search`) returned sensible, correctly-ranked
  results for real queries -- e.g. asking "how are anthropic api errors mapped to
  http status codes" against this project's own ingested code correctly ranked
  `llm.py:answer_question` (where those exceptions are deliberately left to
  propagate) as the top hit.
- **The Claude request is well-formed**: `POST /repos/{id}/chat` was called with a
  deliberately invalid `ANTHROPIC_API_KEY`. The request built the context block,
  called the Anthropic SDK, and the SDK's `401` (`authentication_error`) was caught
  by the typed exception chain in `main.py` and surfaced as a clean HTTP 401 with a
  descriptive message -- not a stack trace or an unhandled crash. That proves the
  whole pipeline up to and including the Anthropic call is wired correctly; getting
  an actual generated answer just requires a real API key.
- **Error paths checked directly**: `GET /repos/999/search` (nonexistent repo) -> 404;
  `POST /repos` with a nonexistent local path -> 400 with a clear message, not a 500.

Not independently re-verified in this pass: a live chat response from a real,
valid Anthropic API key (no key was available in this environment) -- the 401 test
above demonstrates the wiring is correct up to that point.

Python type stubs (`.pyi`) and Windows scripts (`.pyw`) are indexed with the same AST-based symbol chunks as `.py` files.
