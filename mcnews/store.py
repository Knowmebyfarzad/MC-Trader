"""Persistence: raw posts (JSONL), analysis reports (JSON) and read state.

Layout (all under ``data/``):

    data/raw/raw-YYYY_MM_DD.jsonl     one JSON object per fetched post
    data/analyses/analysis-YYYY_MM_DD_HH_MM-<tag>.json
    data/state.json                 {"channels": {"@chan": <last_message_id>}}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import AnalysisReport, NewsItem

STATE_VERSION = 1


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file, creating parents. Returns records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a JSONL file, skipping malformed lines."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


class SeenState:
    """Remembers the newest message id per channel so runs are incremental."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._channels: dict[str, int] = {}
        self.load()

    def load(self) -> "SeenState":
        if not self.path.is_file():
            return self
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self
        channels = data.get("channels") if isinstance(data, dict) else None
        if isinstance(channels, dict):
            clean: dict[str, int] = {}
            for key, value in channels.items():
                try:
                    clean[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            self._channels = clean
        return self

    def get(self, channel: str) -> int | None:
        return self._channels.get(channel)

    def update(self, channel: str, last_message_id: int | None) -> bool:
        """Store ``last_message_id`` if it is newer. Returns True when changed."""
        if not channel or not last_message_id:
            return False
        current = self._channels.get(channel)
        if current is not None and int(last_message_id) <= current:
            return False
        self._channels[channel] = int(last_message_id)
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"version": STATE_VERSION, "channels": dict(sorted(self._channels.items()))}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")


class NewsStore:
    """Filesystem store used by the CLI commands."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.analyses_dir = self.data_dir / "analyses"
        self.state = SeenState(self.data_dir / "state.json")

    # ------------------------------------------------------------------- raw
    def raw_path(self, when: datetime | None = None) -> Path:
        stamp = (when or datetime.now(timezone.utc)).strftime("%Y_%m_%d")
        return self.raw_dir / ("raw-%s.jsonl" % stamp)

    def save_items(self, items: Iterable[NewsItem]) -> tuple[Path, int]:
        path = self.raw_path()
        count = append_jsonl(path, [item.to_dict() for item in items])
        return path, count

    def known_uids(self, days: int = 7, now: datetime | None = None) -> set[str]:
        """UIDs already stored in the last ``days`` raw files (de-duplication)."""
        reference = now or datetime.now(timezone.utc)
        uids: set[str] = set()
        for offset in range(max(1, days)):
            moment = datetime.fromtimestamp(reference.timestamp() - offset * 86400, tz=timezone.utc)
            path = self.raw_dir / ("raw-%s.jsonl" % moment.strftime("%Y_%m_%d"))
            for record in iter_jsonl(path):
                channel = str(record.get("channel") or "")
                message_id = record.get("message_id")
                if channel and message_id is not None:
                    uids.add("%s:%s" % (channel, message_id))
        return uids

    def load_items(self, path: Path | None = None) -> list[NewsItem]:
        """Load NewsItems from a raw file (defaults to the newest one)."""
        target = Path(path) if path else self.latest_raw_path()
        if target is None or not target.is_file():
            return []
        return [NewsItem.from_dict(record) for record in iter_jsonl(target)]

    def latest_raw_path(self) -> Path | None:
        if not self.raw_dir.is_dir():
            return None
        files = sorted(self.raw_dir.glob("raw-*.jsonl"))
        return files[-1] if files else None

    # -------------------------------------------------------------- analyses
    def save_report(self, report: AnalysisReport, tag: str = "") -> Path:
        stamp = report.generated_at.strftime("%Y_%m_%d_%H_%M")
        suffix = ("-" + tag) if tag else ""
        path = self.analyses_dir / ("analysis-%s%s.json" % (stamp, suffix))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
