"""Prompt construction for the local model."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import NewsItem
from .utils import truncate

SYSTEM_PROMPT = (
    "You are a careful news analyst for a short-term trading desk.\n"
    "You receive raw posts from Telegram news channels and turn them into a "
    "compact, factual assessment.\n"
    "Rules:\n"
    "1. Only use facts present in the provided posts. Never invent numbers, "
    "prices, names or events.\n"
    "2. If a post is unclear, promotional, emotional or duplicated, say so "
    "instead of guessing.\n"
    "3. Prefer neutral wording; this is information, not financial advice.\n"
    "4. Always answer with a single JSON object and nothing else - no markdown, "
    "no code fences, no explanation around it."
)

JSON_CONTRACT = """Return exactly this JSON shape:
{
  "overall": {
    "summary": "2-4 sentence digest of the batch",
    "sentiment": "bullish" | "bearish" | "neutral" | "mixed",
    "confidence": 0.0-1.0,
    "market_impact": 1-10,
    "tradable": true,
    "time_horizon": "intraday" | "days" | "weeks" | "months" | "unspecified",
    "key_drivers": ["short phrase", "..."]
  },
  "items": [
    {
      "channel": "channel key exactly as given",
      "message_id": 123,
      "summary": "one sentence",
      "sentiment": "bullish" | "bearish" | "neutral" | "mixed",
      "impact": 1-10,
      "assets": ["BTC", "gold", "..."],
      "topics": ["macro", "..."],
      "actionable": false
    }
  ],
  "watchlist": ["what to keep monitoring"],
  "risks": ["what could make this reading wrong"]
}"""


def format_item(index: int, item: NewsItem, max_chars: int = 1500) -> str:
    """Render one post as a numbered, model-friendly block."""
    body = truncate(item.text, max_chars)
    flags: list[str] = []
    if item.has_media:
        flags.append("media:%s" % (item.media_type or "yes"))
    if item.views is not None:
        flags.append("views:%s" % item.views)
    header = "[%d] channel=%s title=%s message_id=%d date=%s%s" % (
        index,
        item.channel,
        item.label,
        item.message_id,
        item.date.strftime("%Y-%m-%d %H:%M UTC"),
        (" " + " ".join(flags)) if flags else "",
    )
    return "%s\ntext: %s" % (header, body or "<empty>")


def build_analysis_prompt(
    batch: Sequence[NewsItem],
    batch_index: int = 1,
    batch_total: int = 1,
    max_message_chars: int = 1500,
    extra_instructions: str | None = None,
) -> str:
    """User prompt for a batch of posts."""
    blocks = [format_item(i + 1, item, max_message_chars) for i, item in enumerate(batch)]
    header_parts = [
        "Analyse the %d Telegram post(s) below." % len(blocks),
        "Batch %d of %d." % (batch_index, batch_total) if batch_total > 1 else "",
        "Echo back the exact channel key and message_id for every post you score.",
        "Score only posts that carry real information; still list the others with "
        '"sentiment": "neutral" and "actionable": false.',
    ]
    if extra_instructions:
        header_parts.append(extra_instructions.strip())
    header = "\n".join(p for p in header_parts if p)
    return "%s\n\n%s\n\nPOSTS:\n%s" % (header, JSON_CONTRACT, "\n\n".join(blocks))


def build_label(items: Iterable[NewsItem]) -> str:
    """Short human label of the analysed sources (for the report header)."""
    names = []
    for item in items:
        name = item.label or item.channel  # label falls back to the channel key
        if name and name not in names:
            names.append(name)
    return ", ".join(names)
