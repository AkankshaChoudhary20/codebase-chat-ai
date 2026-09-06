"""Calls the Anthropic Claude API with retrieved code chunks as grounding
context, instructing it to cite the specific file:line locations it used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import anthropic

from .config import get_settings
from .retrieval import format_location

SYSTEM_PROMPT = """You are a code-aware assistant answering questions about a specific \
codebase. You are given excerpts retrieved from that codebase -- each labeled with its \
exact file path and line range, e.g. `app/db.py:12-40`.

Rules:
- Answer using only the provided excerpts. If they don't contain enough information to \
answer confidently, say so plainly instead of guessing or inventing code that isn't shown.
- Every claim about how the code works must cite the specific file:line location it came \
from, e.g. "Requests are retried in `app/http_client.py:88-104`."
- If multiple excerpts are relevant, cite each one you actually used.
- Be concise and technical -- the reader is a developer looking at this codebase, not a \
newcomer to programming.
"""


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file "
            "(see .env.example) to enable /chat."
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt."""
    parts = []
    for chunk in chunks:
        location = format_location(chunk)
        label = f"{location}"
        if chunk.get("symbol_name"):
            label += f" ({chunk['symbol_type']} {chunk['symbol_name']})"
        parts.append(f"--- {label} ---\n{chunk['code_text']}")
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[dict[str, Any]]) -> str:
    """Ask Claude to answer `question` using `chunks` as the only context.

    Returns the model's text answer. Raises `RuntimeError` if no API key is
    configured. Anthropic SDK exceptions (`AuthenticationError`,
    `RateLimitError`, `APIStatusError`, `APIConnectionError`, ...) are
    deliberately left to propagate uncaught -- main.py catches them as a
    typed chain and maps each to a clean HTTP status instead of a 500 stack
    trace. This is what lets a missing/invalid API key surface as a clean
    401, proving the request was well-formed and actually reached Anthropic.
    """
    settings = get_settings()
    client = get_client()

    if not chunks:
        context_block = "(No relevant code excerpts were found in this repo.)"
    else:
        context_block = build_context_block(chunks)

    user_message = (
        f"Codebase excerpts:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the excerpts above, citing the exact "
        "file:line location(s) you relied on."
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    if response.stop_reason == "refusal":
        return "The model declined to answer this question."

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts).strip() or "(The model returned an empty response.)"
