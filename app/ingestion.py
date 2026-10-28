"""Repo ingestion: resolve a local path or git URL, walk the tree, chunk each
source file by function/class boundary, embed the chunks locally, and store
them in pgvector.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import db
from .chunking import chunk_file
from .config import get_settings
from .embeddings import embed_texts

# Directories that are never worth indexing: VCS internals, dependency
# trees, and build output. Matched by directory *name* anywhere in the tree.
SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",  # Rust / Java build output
    ".idea",
    ".vscode",
    "vendor",  # Go vendored deps
    ".mypy_cache",
    ".pytest_cache",
    "egg-info",
}

# Extensions we know how to chunk meaningfully, plus a handful of common
# plain-text source/config extensions that fall back to whole-file chunks.
# Everything else is skipped outright (images, archives, etc. are also
# caught by the binary-content sniff below, but this avoids wasting time on
# obviously irrelevant files).
INDEXABLE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".sql",
    ".sh",
}

_GIT_URL_PATTERN = re.compile(r"^(https?://|git@|git://|ssh://)", re.IGNORECASE)


def is_git_url(source: str) -> bool:
    return bool(_GIT_URL_PATTERN.match(source.strip())) or source.strip().endswith(".git")


def _looks_binary(sample: bytes) -> bool:
    """Cheap binary sniff: a NUL byte anywhere in the first chunk almost
    always means "not text". This is the same heuristic `git` itself uses.
    """
    return b"\x00" in sample


@dataclass
class IngestResult:
    repo_id: int
    name: str
    source: str
    files_scanned: int
    files_indexed: int
    chunks_created: int


def clone_repo(git_url: str, workdir: str) -> Path:
    """`git clone --depth 1` the given URL into a fresh subdirectory of
    `workdir`. Raises RuntimeError with git's own stderr on failure (e.g.
    bad URL, auth failure, network error) so the API can surface it as a
    clean 400 rather than a stack trace.
    """
    base = Path(workdir)
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"{uuid.uuid4().hex[:12]}"

    proc = subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return dest


def resolve_source(source: str, workdir: str) -> tuple[Path, str]:
    """Resolve `source` (a local filesystem path or a git URL) to a local
    directory to walk. Returns (path, display_name).
    """
    source = source.strip()
    if is_git_url(source):
        path = clone_repo(source, workdir)
        name = Path(source.rstrip("/").removesuffix(".git")).name or source
        return path, name

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Local path does not exist: {path}")
    if not path.is_dir():
        raise RuntimeError(f"Local path is not a directory: {path}")
    return path, path.name


def iter_source_files(root: Path, max_file_size_bytes: int):
    """Yield (relative_path, source_text) for every file under `root` worth
    indexing: skips VCS/dependency/build directories, binary files, files
    over the size cap, and files whose bytes don't decode as UTF-8 text.
    """
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        # Prune dependency trees before traversing them, rather than filtering
        # their files after an expensive recursive walk.
        subdirectories[:] = sorted(name for name in subdirectories
            if name not in SKIP_DIR_NAMES and not name.endswith('.egg-info')
            and not (Path(directory) / name).is_symlink())
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.is_symlink() or path.suffix.lower() not in INDEXABLE_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
                if size == 0 or size > max_file_size_bytes:
                    continue
                with path.open('rb') as source_file:
                    raw = source_file.read(max_file_size_bytes + 1)
            except OSError:
                continue
            if len(raw) > max_file_size_bytes or _looks_binary(raw[:8192]):
                continue
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                continue
            yield path.relative_to(root).as_posix(), text


def ingest_repo(source: str, name: str | None = None) -> IngestResult:
    """End-to-end ingestion: resolve source -> walk files -> chunk -> embed
    -> store. This is the implementation behind `POST /repos`.
    """
    settings = get_settings()
    root, resolved_name = resolve_source(source, settings.clone_workdir)
    display_name = name or resolved_name

    repo_id = db.insert_repo(display_name, source)

    files_scanned = 0
    files_indexed = 0
    all_chunk_dicts: list[dict] = []
    pending_texts: list[str] = []
    pending_chunks = []

    def flush():
        nonlocal pending_texts, pending_chunks
        if not pending_chunks:
            return
        vectors = embed_texts(pending_texts)
        for chunk, vector in zip(pending_chunks, vectors):
            all_chunk_dicts.append(
                {
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "code_text": chunk.code_text,
                    "embedding": vector,
                }
            )
        pending_texts = []
        pending_chunks = []

    BATCH_SIZE = 64

    for rel_path, text in iter_source_files(root, settings.max_file_size_bytes):
        files_scanned += 1
        chunks = chunk_file(rel_path, text)
        if not chunks:
            continue
        files_indexed += 1
        for chunk in chunks:
            pending_chunks.append(chunk)
            pending_texts.append(chunk.embedding_text())
            if len(pending_chunks) >= BATCH_SIZE:
                flush()
    flush()

    chunks_created = db.insert_code_chunks(repo_id, all_chunk_dicts)

    return IngestResult(
        repo_id=repo_id,
        name=display_name,
        source=source,
        files_scanned=files_scanned,
        files_indexed=files_indexed,
        chunks_created=chunks_created,
    )
