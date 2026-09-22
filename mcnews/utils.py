"""Small shared helpers: text hygiene, truncation and budget based chunking."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

_INLINE_WS = re.compile(r"[ \t\u00a0\u200b]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

TRUE_WORDS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_WORDS = {"0", "false", "no", "n", "off", "disabled"}


def clean_text(text: str | None) -> str:
    """Normalise whitespace and strip control characters from message text."""
    if not text:
        return ""
    cleaned = _CONTROL.sub("", str(text).replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = _INLINE_WS.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return "\n".join(line.strip() for line in cleaned.split("\n")).strip()


def truncate(text: str, max_chars: int, suffix: str = " ...[truncated]") -> str:
    """Character-safe truncation (never raises on ``max_chars`` <= 0)."""
    if max_chars <= 0:
        return suffix.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def shorten(text: str, max_chars: int) -> str:
    """Truncate with a plain ellipsis, for display only."""
    text = text.replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in TRUE_WORDS:
        return True
    if token in FALSE_WORDS:
        return False
    return default


def chunk_seq(items: Sequence[T], size_of: Callable[[T], int], budget: int) -> list[list[T]]:
    """Group items so each group's total size stays under ``budget``.

    A single item larger than ``budget`` gets a chunk of its own (it is never
    dropped or split here - callers should truncate beforehand).
    """
    chunks: list[list[T]] = []
    current: list[T] = []
    current_size = 0
    for item in items:
        size = max(0, int(size_of(item)))
        if current and current_size + size > budget:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def as_utc(value: datetime | None) -> datetime:
    """Return a timezone aware UTC datetime (naive input is assumed UTC)."""
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def fmt_dt(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if value is None:
        return "-"
    return as_utc(value).strftime(fmt)


def dedupe(items: Iterable[T], key: Callable[[T], str]) -> list[T]:
    """Stable de-duplication preserving input order."""
    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        token = key(item)
        if token in seen:
            continue
        seen.add(token)
        out.append(item)
    return out


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({v for v in (str(x).strip() for x in values) if v})


def coerce_int(value: object, default: int | None = None, minimum: int | None = None,
               maximum: int | None = None) -> int | None:
    """Best-effort int conversion (used for LLM output that may be sloppy)."""
    if value is None or isinstance(value, bool):
        result: int | None = default
    elif isinstance(value, (int, float)):
        result = int(value)
    else:
        token = str(value).strip()
        try:
            result = int(float(token))
        except ValueError:
            result = default
    if result is None:
        return None
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(value: object, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def coerce_str_list(value: object) -> list[str]:
    """Normalise ``["a", "b"]``, ``"a, b"`` and ``None`` into a clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for entry in value:
            parts.extend(str(entry).replace(";", ",").split(","))
    else:
        parts = [str(value)]
    return unique_sorted(part.strip() for part in parts if str(part).strip())


def coerce_sentence_list(value: object) -> list[str]:
    """Like :func:`coerce_str_list` but keeps sentences intact.

    Used for free-text items (risks, watchlist) where commas are normal
    punctuation rather than separators.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [line for chunk in value.split(";") for line in chunk.split("\n")]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(entry) for entry in value]
    else:
        parts = [str(value)]
    cleaned: list[str] = []
    for part in parts:
        text = " ".join(str(part).split()).strip(" -")
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
