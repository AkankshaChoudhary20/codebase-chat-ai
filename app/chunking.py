"""Code-aware chunking: split source files by function/class/method boundary
instead of a fixed-size sliding window.

Strategy, by file type:

- **Python (`.py`)**: parsed with the stdlib `ast` module. This gives exact,
  syntax-aware boundaries (`node.lineno` / `node.end_lineno`) for every
  top-level function, class, and method. This is the one language where
  chunk boundaries are guaranteed correct (modulo the file parsing at all).

- **JavaScript / TypeScript / Go / Java / Rust**: no AST available without
  adding a heavy per-language parser dependency, so these use a regex +
  brace-counting heuristic instead: a line-by-line regex scan finds
  function/method/class/struct declarations, then a simple `{`/`}` depth
  counter finds where each block ends. This is genuinely a heuristic, not a
  parser, and it has known failure modes -- see the README for the honest
  list (braces inside strings/comments, multi-line signatures, arrow
  functions without a block body, etc). It works well on typical,
  conventionally-formatted source but will occasionally mis-split or miss a
  symbol on unusual formatting.

- **Everything else**: if a file's extension isn't recognized, or the
  language-specific pass finds no symbols (e.g. a config file, a script with
  only top-level statements, or a parse error), the whole file becomes a
  single chunk with `symbol_type="module"`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

SymbolType = str  # "function" | "class" | "method" | "module"


@dataclass
class CodeChunk:
    file_path: str
    symbol_name: str | None
    symbol_type: SymbolType
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    code_text: str

    def embedding_text(self) -> str:
        """Text actually sent to the embedding model.

        Prepending the file path and symbol gives the embedding a little
        extra signal about *what* the code is (e.g. matches "the retry
        logic in http_client" even if the identifier is named `_do_request`)
        without polluting the stored `code_text` used for display/citation.
        """
        header = f"# {self.file_path}"
        if self.symbol_name:
            header += f" :: {self.symbol_type} {self.symbol_name}"
        return f"{header}\n{self.code_text}"


# ---------------------------------------------------------------------------
# Python: real AST-based chunking
# ---------------------------------------------------------------------------


def chunk_python_source(source: str, file_path: str) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # caller falls back to whole-file-as-one-chunk

    lines = source.splitlines()

    def line_span(node: ast.AST) -> tuple[int, int]:
        start = node.lineno  # type: ignore[attr-defined]
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            start = min(start, min(d.lineno for d in decorators))
        end = getattr(node, "end_lineno", start)
        return start, end

    def extract(start: int, end: int) -> str:
        return "\n".join(lines[start - 1 : end])

    chunks: list[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = line_span(node)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=start,
                    end_line=end,
                    code_text=extract(start, end),
                )
            )
        elif isinstance(node, ast.ClassDef):
            start, end = line_span(node)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=start,
                    end_line=end,
                    code_text=extract(start, end),
                )
            )
            # Also emit each method as its own finer-grained chunk, so a
            # search for a specific method finds a tight, precise citation
            # in addition to the whole-class chunk above.
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_start, m_end = line_span(sub)
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            symbol_name=f"{node.name}.{sub.name}",
                            symbol_type="method",
                            start_line=m_start,
                            end_line=m_end,
                            code_text=extract(m_start, m_end),
                        )
                    )

    if not chunks:
        return []  # Preserve whole-file fallback for modules without symbols.

    # Keep imports, constants and executable module statements searchable
    # when the same file also contains function or class definitions.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = line_span(node)
            chunks.append(CodeChunk(file_path, None, "module", start, end, extract(start, end)))
    # Preserve standalone comments and file headers outside AST node spans.
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    start = 0
    while start < len(lines):
        if start + 1 in covered:
            start += 1
            continue
        end = start + 1
        while end < len(lines) and end + 1 not in covered:
            end += 1
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(CodeChunk(file_path, None, "module", start + 1, end, text))
        start = end
    chunks.sort(key=lambda chunk: chunk.start_line)
    return chunks


# ---------------------------------------------------------------------------
# Heuristic regex + brace-counting chunking for other languages
# ---------------------------------------------------------------------------

_IDENT = r"[A-Za-z_$][\w$]*"

_JS_TS_TOP_LEVEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>{_IDENT})"),
        "class",
    ),
    (
        re.compile(
            rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(?P<name>{_IDENT})\s*\("
        ),
        "function",
    ),
    (
        # const foo = (args) => { ... }   /  const foo = async (args): T => { ... }
        re.compile(
            rf"^\s*(?:export\s+)?const\s+(?P<name>{_IDENT})\s*(?::\s*[^=]+?)?=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]+?)?=>"
        ),
        "function",
    ),
    (
        # const foo = function (args) { ... }
        re.compile(rf"^\s*(?:export\s+)?const\s+(?P<name>{_IDENT})\s*=\s*(?:async\s+)?function\s*\*?\s*\("),
        "function",
    ),
]

# Indented method inside a class body: optional modifiers, identifier, "(...)"
# then either "{" or ": ReturnType {". Deliberately requires >=2 leading
# spaces (i.e. it's nested inside something) to avoid matching top-level
# function calls / control-flow statements.
_JS_TS_METHOD_PATTERN = re.compile(
    r"^[ \t]{2,}(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+|get\s+|set\s+|\*\s*)*"
    rf"(?P<name>{_IDENT})\s*\([^;{{]*\)\s*(?::\s*[^{{=]+)?\{{"
)
_JS_TS_METHOD_EXCLUDE = {"if", "for", "while", "switch", "catch", "else", "function"}

_GO_TYPE_PATTERN = re.compile(r"^type\s+(?P<name>[A-Za-z_]\w*)\s+(?:struct|interface)\s*\{")
_GO_FUNC_PATTERN = re.compile(r"^func\s*(?:\((?P<receiver>[^)]*)\))?\s*(?P<name>[A-Za-z_]\w*)\s*\(")

_JAVA_CLASS_PATTERN = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?"
    r"(?:final\s+|abstract\s+)?(?:class|interface|enum)\s+(?P<name>[A-Za-z_]\w*)"
)
_JAVA_METHOD_PATTERN = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:public|private|protected)\s+"
    r"(?:static\s+|final\s+|abstract\s+|synchronized\s+|default\s+)*"
    r"[\w<>\[\],.\s]+?\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:throws\s+[\w,.\s<>]+)?\s*[{;]"
)
_JAVA_METHOD_EXCLUDE = {"if", "for", "while", "switch", "catch", "new", "return"}

_RUST_STRUCT_PATTERN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(?P<name>[A-Za-z_]\w*)")
_RUST_FN_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>[A-Za-z_]\w*)"
)


def _find_block_end(lines: list[str], start_idx: int, max_lookahead: int = 5000) -> int:
    """Given the 0-indexed line where a declaration starts, find the 1-indexed
    end line of its `{ ... }` block by counting brace depth.

    Heuristic, not a parser: braces inside string literals, template
    literals, char literals, or comments are counted like any other brace.
    In practice this rarely matters for well-formatted source (braces
    inside strings are uncommon), but it's a known, documented limitation.

    If the declaration line ends in `;` before any `{` is seen (e.g. an
    interface method signature or an abstract method with no body), the
    declaration is treated as a single-line chunk.
    """
    depth = 0
    seen_brace = False
    limit = min(len(lines), start_idx + max_lookahead)
    end_idx = start_idx

    for i in range(start_idx, limit):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                seen_brace = True
            elif ch == "}":
                depth -= 1
        end_idx = i
        if seen_brace and depth <= 0:
            return end_idx + 1
        if not seen_brace and i == start_idx and ";" in line and "{" not in line:
            return start_idx + 1

    # No closing brace found in the lookahead window (or EOF) -- return
    # whatever we scanned rather than looping forever on a malformed file.
    return end_idx + 1


def chunk_generic_source(source: str, file_path: str, ext: str) -> list[CodeChunk]:
    lines = source.splitlines()
    matches: list[tuple[int, str, str]] = []  # (0-indexed line, name, symbol_type)

    if ext in (".js", ".jsx", ".ts", ".tsx"):
        for i, line in enumerate(lines):
            matched = False
            for pattern, sym_type in _JS_TS_TOP_LEVEL_PATTERNS:
                m = pattern.match(line)
                if m:
                    matches.append((i, m.group("name"), sym_type))
                    matched = True
                    break
            if matched:
                continue
            m = _JS_TS_METHOD_PATTERN.match(line)
            if m and m.group("name") not in _JS_TS_METHOD_EXCLUDE:
                matches.append((i, m.group("name"), "method"))

    elif ext == ".go":
        for i, line in enumerate(lines):
            m = _GO_TYPE_PATTERN.match(line)
            if m:
                matches.append((i, m.group("name"), "class"))
                continue
            m = _GO_FUNC_PATTERN.match(line)
            if m:
                receiver = m.group("receiver")
                name = m.group("name")
                if receiver:
                    type_match = re.search(r"([A-Za-z_]\w*)\s*$", receiver.replace("*", "").strip())
                    recv_type = type_match.group(1) if type_match else receiver.strip()
                    matches.append((i, f"{recv_type}.{name}", "method"))
                else:
                    matches.append((i, name, "function"))

    elif ext == ".java":
        for i, line in enumerate(lines):
            m = _JAVA_CLASS_PATTERN.match(line)
            if m:
                matches.append((i, m.group("name"), "class"))
                continue
            m = _JAVA_METHOD_PATTERN.match(line)
            if m and m.group("name") not in _JAVA_METHOD_EXCLUDE:
                matches.append((i, m.group("name"), "method"))

    elif ext == ".rs":
        for i, line in enumerate(lines):
            m = _RUST_STRUCT_PATTERN.match(line)
            if m:
                matches.append((i, m.group("name"), "class"))
                continue
            m = _RUST_FN_PATTERN.match(line)
            if m:
                sym_type = "method" if m.group("indent") else "function"
                matches.append((i, m.group("name"), sym_type))

    chunks: list[CodeChunk] = []
    seen_spans: set[tuple[int, int]] = set()
    for line_idx, name, sym_type in matches:
        start_line = line_idx + 1
        end_line = _find_block_end(lines, line_idx)
        span = (start_line, end_line)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        chunks.append(
            CodeChunk(
                file_path=file_path,
                symbol_name=name,
                symbol_type=sym_type,
                start_line=start_line,
                end_line=end_line,
                code_text="\n".join(lines[start_line - 1 : end_line]),
            )
        )

    if chunks:
        # Symbols alone omit imports, constants and top-level statements.
        # Emit contiguous uncovered spans with their original line numbers.
        covered = [False] * len(lines)
        for chunk in chunks:
            covered[chunk.start_line - 1:chunk.end_line] = [True] * (chunk.end_line - chunk.start_line + 1)
        start = 0
        while start < len(lines):
            if covered[start]:
                start += 1
                continue
            end = start + 1
            while end < len(lines) and not covered[end]:
                end += 1
            text = "\n".join(lines[start:end])
            if text.strip():
                chunks.append(CodeChunk(file_path, None, "module", start + 1, end, text))
            start = end
        chunks.sort(key=lambda chunk: chunk.start_line)
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PYTHON_EXTS = {".py", ".pyi", ".pyw"}
_GENERIC_EXTS = {".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs"}


def chunk_file(file_path: str, source: str) -> list[CodeChunk]:
    """Split one file's source into chunks. Never raises on bad/unknown
    input -- falls back to a single whole-file chunk.
    """
    ext = Path(file_path).suffix.lower()
    chunks: list[CodeChunk] = []

    if ext in _PYTHON_EXTS:
        chunks = chunk_python_source(source, file_path)
    elif ext in _GENERIC_EXTS:
        chunks = chunk_generic_source(source, file_path, ext)

    if not chunks:
        lines = source.splitlines()
        if not lines:
            return []
        chunks = [
            CodeChunk(
                file_path=file_path,
                symbol_name=Path(file_path).name,
                symbol_type="module",
                start_line=1,
                end_line=len(lines),
                code_text=source,
            )
        ]

    return chunks
