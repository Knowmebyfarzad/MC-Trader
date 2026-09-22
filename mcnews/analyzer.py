"""Turn fetched posts into a structured report using the local model."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Sequence

from .llm import JSONExtractionError, LocalLLMClient, extract_json
from .models import AnalysisReport, NewsItem
from .prompts import SYSTEM_PROMPT, build_analysis_prompt, build_label
from .utils import (
    chunk_seq,
    coerce_float,
    coerce_int,
    coerce_sentence_list,
    coerce_str_list,
    dedupe,
    parse_bool,
    unique_sorted,
)

logger = logging.getLogger(__name__)

SENTIMENTS = ("bullish", "bearish", "neutral", "mixed")
HORIZONS = ("intraday", "days", "weeks", "months", "unspecified")

SENTIMENT_ALIASES = {
    "bull": "bullish",
    "bullish": "bullish",
    "up": "bullish",
    "positive": "bullish",
    "long": "bullish",
    "bear": "bearish",
    "bearish": "bearish",
    "down": "bearish",
    "negative": "bearish",
    "short": "bearish",
    "neutral": "neutral",
    "flat": "neutral",
    "mixed": "mixed",
    "unclear": "mixed",
    "unknown": "mixed",
}

HORIZON_ALIASES = {
    "hour": "intraday",
    "hours": "intraday",
    "hourly": "intraday",
    "intra-day": "intraday",
    "short": "intraday",
    "short-term": "days",
    "day": "days",
    "daily": "days",
    "week": "weeks",
    "weekly": "weeks",
    "month": "months",
    "monthly": "months",
    "long": "months",
    "long-term": "months",
    "quarter": "months",
    "unknown": "unspecified",
    "unclear": "unspecified",
    "n/a": "unspecified",
    "none": "unspecified",
}

MAX_LIST_ITEMS = 20


def normalize_sentiment(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = SENTIMENT_ALIASES.get(token, token)
    return token if token in SENTIMENTS else "mixed"


def normalize_horizon(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "unspecified"
    if token in HORIZONS:
        return token
    if token in HORIZON_ALIASES:
        return HORIZON_ALIASES[token]
    for known in ("intraday", "days", "weeks", "months"):
        if token.startswith(known) or known.startswith(token):
            return known
    return "unspecified"


def normalize_overall(overall: dict[str, Any]) -> dict[str, Any]:
    """Clamp one batch ``overall`` block into a predictable shape."""
    confidence = coerce_float(overall.get("confidence"), 0.5)
    if confidence is None:
        confidence = 0.5
    return {
        "summary": str(overall.get("summary") or "").strip(),
        "sentiment": normalize_sentiment(overall.get("sentiment")),
        "confidence": round(min(1.0, max(0.0, confidence)), 2),
        "market_impact": coerce_int(overall.get("market_impact"), 1, 1, 10) or 1,
        "tradable": parse_bool(overall.get("tradable"), False),
        "time_horizon": normalize_horizon(overall.get("time_horizon")),
        "key_drivers": coerce_str_list(overall.get("key_drivers"))[:MAX_LIST_ITEMS],
    }


def normalize_item(entry: Any, channel_hint: str = "") -> dict[str, Any] | None:
    """Clamp one per-post analysis entry. Returns ``None`` when unusable."""
    if not isinstance(entry, dict):
        return None
    summary = str(entry.get("summary") or entry.get("gist") or "").strip()
    channel = str(entry.get("channel") or channel_hint or "").strip()
    message_id = coerce_int(entry.get("message_id"), None)
    if not summary and not message_id:
        return None
    return {
        "channel": channel,
        "message_id": message_id,
        "summary": summary,
        "sentiment": normalize_sentiment(entry.get("sentiment")),
        "impact": coerce_int(entry.get("impact"), 1, 1, 10) or 1,
        "assets": coerce_str_list(entry.get("assets"))[:MAX_LIST_ITEMS],
        "topics": coerce_str_list(entry.get("topics"))[:MAX_LIST_ITEMS],
        "actionable": parse_bool(entry.get("actionable"), False),
    }


def merge_overalls(overalls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Deterministically combine per-batch overviews (no extra model call).

    Defensive on purpose: it also accepts raw (unnormalised) model output.
    """
    if not overalls:
        return {}
    if len(overalls) == 1:
        return dict(overalls[0])

    votes = Counter(normalize_sentiment(overall.get("sentiment")) for overall in overalls)
    top_count = max(votes.values())
    winners = [name for name, count in votes.items() if count == top_count]
    sentiment = winners[0] if len(winners) == 1 else "mixed"

    impacts = [
        value
        for value in (coerce_int(overall.get("market_impact"), None, 1, 10) for overall in overalls)
        if value is not None
    ]
    confidences = [
        value for value in (coerce_float(overall.get("confidence")) for overall in overalls) if value is not None
    ]
    horizons = Counter(
        normalize_horizon(overall.get("time_horizon"))
        for overall in overalls
        if str(overall.get("time_horizon") or "").strip()
    )

    summaries: list[str] = []
    for overall in overalls:
        text = str(overall.get("summary") or "").strip()
        if text and text not in summaries:
            summaries.append(text)

    drivers: list[str] = []
    for overall in overalls:
        drivers.extend(coerce_str_list(overall.get("key_drivers")))

    return {
        "summary": " | ".join(summaries),
        "sentiment": sentiment,
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.5,
        "market_impact": max(impacts) if impacts else 1,
        "tradable": any(parse_bool(overall.get("tradable"), False) for overall in overalls),
        "time_horizon": horizons.most_common(1)[0][0] if horizons else "unspecified",
        "key_drivers": unique_sorted(drivers)[:MAX_LIST_ITEMS],
        "batches_merged": len(overalls),
    }

def synthesize_overall_from_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fallback overview when the model scored posts but omitted ``overall``.

    Small local models occasionally skip the ``overall`` block; rather than
    showing "n/a" we derive a conservative overview from the per-post scores.
    """
    if not items:
        return {}
    votes = Counter(normalize_sentiment(entry.get("sentiment")) for entry in items)
    ranked = votes.most_common()
    sentiment = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else "mixed"
    impacts = [
        value
        for value in (coerce_int(entry.get("impact"), None, 1, 10) for entry in items)
        if value is not None
    ]
    return {
        "summary": "The model returned no 'overall' block, so this overview is derived from the per-post scores.",
        "sentiment": sentiment,
        "confidence": 0.3,
        "market_impact": max(impacts) if impacts else 1,
        "tradable": any(parse_bool(entry.get("actionable"), False) for entry in items),
        "time_horizon": "unspecified",
        "key_drivers": [],
        "derived": True,
    }


class NewsAnalyzer:
    """Batch posts, call the local model and merge the answers."""

    def __init__(
        self,
        llm: LocalLLMClient,
        batch_chars: int = 6000,
        max_message_chars: int = 1500,
        extra_instructions: str | None = None,
        max_items: int | None = None,
    ) -> None:
        self.llm = llm
        self.batch_chars = max(500, int(batch_chars))
        self.max_message_chars = max(100, int(max_message_chars))
        self.extra_instructions = extra_instructions
        self.max_items = max_items

    # ------------------------------------------------------------------ input
    def prepare(self, items: Iterable[NewsItem], skip_uids: set[str] | None = None) -> list[NewsItem]:
        """Keep usable, de-duplicated posts (newest first)."""
        skipped = skip_uids or set()
        usable = [i for i in items if i.is_usable() and i.uid not in skipped]
        ordered = sorted(usable, key=lambda i: (i.date, i.message_id), reverse=True)
        return dedupe(ordered, key=lambda i: i.uid)

    # --------------------------------------------------------------- analysis
    def analyze(self, items: Iterable[NewsItem], note: str | None = None) -> AnalysisReport:
        prepared = self.prepare(items)
        if self.max_items:
            prepared = prepared[: self.max_items]

        report = AnalysisReport(
            provider=self.llm.provider,
            model=self.llm.model,
            item_count=len(prepared),
            sources=[build_label(prepared)] if prepared else [],
        )
        if not prepared:
            report.overall = {
                "summary": "No usable posts to analyse.",
                "sentiment": "neutral",
                "market_impact": 1,
                "tradable": False,
                "confidence": 0.0,
                "time_horizon": "unspecified",
                "key_drivers": [],
            }
            return report

        batches = chunk_seq(prepared, lambda item: len(item.text) + 200, self.batch_chars)
        report.batch_count = len(batches)
        overalls: list[dict[str, Any]] = []
        watchlist: list[str] = []
        risks: list[str] = []

        for index, batch in enumerate(batches, start=1):
            prompt = build_analysis_prompt(
                batch,
                batch_index=index,
                batch_total=len(batches),
                max_message_chars=self.max_message_chars,
                extra_instructions=self.extra_instructions,
            )
            logger.debug("analysing batch %d/%d (%d posts)", index, len(batches), len(batch))
            raw = self.llm.chat(SYSTEM_PROMPT, prompt)  # LLMUnavailable propagates
            try:
                payload = extract_json(raw)
            except JSONExtractionError as exc:
                logger.warning("batch %d: model answer was not JSON (%s)", index, exc)
                report.failed_batches += 1
                report.raw_responses.append("[unparsed batch %d] %s" % (index, raw[:2000]))
                continue

            if not isinstance(payload, dict):
                logger.warning("batch %d: JSON was not an object", index)
                report.failed_batches += 1
                report.raw_responses.append("[unexpected JSON batch %d] %s" % (index, str(payload)[:2000]))
                continue

            report.parsed_batches += 1
            report.raw_responses.append("batch %d: %s" % (index, str(payload)[:2000]))

            overall = payload.get("overall")
            if isinstance(overall, dict):
                normalized_overall = normalize_overall(overall)
                overalls.append(normalized_overall)
                report.batch_overviews.append(normalized_overall)

            channel_hint = batch[0].channel if len(batch) == 1 else ""
            for entry in payload.get("items") or []:
                normalized = normalize_item(entry, channel_hint=channel_hint)
                if normalized is None:
                    continue
                normalized["batch"] = index
                report.items.append(normalized)

            watchlist.extend(coerce_sentence_list(payload.get("watchlist")))
            risks.extend(coerce_sentence_list(payload.get("risks")))

        report.overall = merge_overalls(overalls)
        if not report.overall and report.items:
            # small models sometimes skip the whole "overall" block
            report.overall = synthesize_overall_from_items(report.items)
        report.watchlist = dedupe(watchlist, key=lambda value: value)[:MAX_LIST_ITEMS]
        report.risks = dedupe(risks, key=lambda value: value)[:MAX_LIST_ITEMS]
        if note:
            report.overall.setdefault("notes", note)
        return report


def attach_context(items: Iterable[NewsItem]) -> dict[str, NewsItem]:
    """Map ``uid -> NewsItem`` so CLI output can add dates and permalinks."""
    return {item.uid: item for item in items}


def report_summary_line(report: AnalysisReport, when: datetime | None = None) -> str:
    overall = report.overall or {}
    stamp = (when or report.generated_at).strftime("%Y-%m-%d %H:%M UTC")
    return "[%s] %s | sentiment=%s impact=%s/10 tradable=%s horizon=%s posts=%d batches=%d (parsed %d)" % (
        stamp,
        report.model,
        overall.get("sentiment", "n/a"),
        overall.get("market_impact", "?"),
        overall.get("tradable", "?"),
        overall.get("time_horizon", "?"),
        report.item_count,
        report.batch_count,
        report.parsed_batches,
    )
