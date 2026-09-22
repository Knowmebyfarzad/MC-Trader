"""Data containers shared by the Telegram reader, the analyzer and the store.

``NewsItem.from_message`` is deliberately duck-typed: the tests feed it plain
fake objects, so the fetching logic can be verified without a Telegram account.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .utils import as_utc, clean_text

PERMALINK_TEMPLATE = "https://t.me/{username}/{message_id}"


def permalink_for(username: str | None, message_id: int) -> str | None:
    """Public post link, only possible for channels with a @username."""
    if not username:
        return None
    return PERMALINK_TEMPLATE.format(username=username.lstrip("@"), message_id=message_id)


@dataclass
class NewsItem:
    """One Telegram post (the unit of news we analyse)."""

    channel: str  # stable key: "@username" or "-1001234567890"
    message_id: int
    text: str
    date: datetime = field(default_factory=lambda: as_utc(None))
    channel_title: str = ""
    channel_username: str | None = None
    has_media: bool = False
    media_type: str | None = None
    views: int | None = None
    forwards: int | None = None
    permalink: str | None = None

    def __post_init__(self) -> None:
        self.text = clean_text(self.text)
        self.date = as_utc(self.date)
        self.channel = (self.channel or "").strip()
        self.channel_title = (self.channel_title or "").strip() or self.channel
        if self.channel_username:
            self.channel_username = self.channel_username.lstrip("@")
        if self.permalink is None:
            self.permalink = permalink_for(self.channel_username, self.message_id)

    # ------------------------------------------------------------------ extras
    @property
    def uid(self) -> str:
        """Unique key used for de-duplication across runs."""
        return "%s:%s" % (self.channel, self.message_id)

    @property
    def label(self) -> str:
        return self.channel_title or self.channel or "unknown"

    def is_usable(self, min_chars: int = 3) -> bool:
        return len(self.text) >= min_chars

    def label_line(self) -> str:
        parts = [self.label, "#%d" % self.message_id, self.date.strftime("%Y-%m-%d %H:%M UTC")]
        if self.permalink:
            parts.append(self.permalink)
        return " | ".join(parts)

    # ------------------------------------------------------------ (de)serialize
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["date"] = self.date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NewsItem":
        payload = dict(data)
        raw_date = payload.pop("date", None)
        if isinstance(raw_date, str) and raw_date:
            try:
                payload["date"] = datetime.fromisoformat(raw_date)
            except ValueError:
                payload["date"] = as_utc(None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_message(
        cls,
        message: Any,
        channel: str,
        channel_title: str = "",
        channel_username: str | None = None,
    ) -> "NewsItem":
        """Build a NewsItem from a Telethon Message (or any look-alike object)."""
        text = (
            getattr(message, "message", None)
            or getattr(message, "text", None)
            or getattr(message, "raw_text", None)
            or ""
        )
        media = getattr(message, "media", None)
        date = getattr(message, "date", None)
        return cls(
            channel=channel,
            message_id=int(getattr(message, "id", 0) or 0),
            text=str(text),
            date=as_utc(date) if isinstance(date, datetime) else as_utc(None),
            channel_title=channel_title,
            channel_username=channel_username,
            has_media=media is not None,
            media_type=type(media).__name__ if media is not None else None,
            views=getattr(message, "views", None),
            forwards=getattr(message, "forwards", None),
        )


@dataclass
class AnalysisReport:
    """Merged result of one analysis run (one or more LLM calls)."""

    provider: str
    model: str
    generated_at: datetime = field(default_factory=lambda: as_utc(None))
    item_count: int = 0
    batch_count: int = 0
    parsed_batches: int = 0
    failed_batches: int = 0
    overall: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    watchlist: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    batch_overviews: list[dict[str, Any]] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisReport":
        payload = dict(data)
        raw_date = payload.pop("generated_at", None)
        if isinstance(raw_date, str) and raw_date:
            try:
                payload["generated_at"] = datetime.fromisoformat(raw_date)
            except ValueError:
                payload["generated_at"] = as_utc(None)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
