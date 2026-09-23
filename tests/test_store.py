"""Tests for the JSONL raw store, the analysis store and the read state."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcnews.models import AnalysisReport, NewsItem
from mcnews.store import NewsStore, SeenState, append_jsonl, iter_jsonl


def make_item(message_id: int, channel: str = "c", text: str = "hello world") -> NewsItem:
    return NewsItem(
        channel=channel,
        message_id=message_id,
        text=text,
        date=datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=message_id),
        channel_title="Desk",
        channel_username="desk",
    )


class JsonlTests(unittest.TestCase):
    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "raw.jsonl"
            written = append_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(written, 2)
            self.assertEqual(list(iter_jsonl(path)), [{"a": 1}, {"b": 2}])

    def test_broken_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text('{"a": 1}\nnot json\n\n{"b": 2}\n[1,2]\n', encoding="utf-8")
            self.assertEqual(list(iter_jsonl(path)), [{"a": 1}, {"b": 2}])

    def test_missing_file_yields_nothing(self):
        self.assertEqual(list(iter_jsonl(Path("nope.jsonl"))), [])


class SeenStateTests(unittest.TestCase):
    def test_update_keeps_the_newest_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = SeenState(Path(tmp) / "state.json")
            self.assertIsNone(state.get("@a"))
            self.assertTrue(state.update("@a", 10))
            self.assertFalse(state.update("@a", 9))
            self.assertFalse(state.update("@a", 10))
            self.assertTrue(state.update("@a", 11))
            self.assertEqual(state.get("@a"), 11)

    def test_update_ignores_falsy_ids(self):
        state = SeenState(Path("unused-state.json"))
        self.assertFalse(state.update("@a", None))
        self.assertFalse(state.update("@a", 0))
        self.assertFalse(state.update("", 5))
        self.assertEqual(state.as_dict()["channels"], {})

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = SeenState(path)
            state.update("@b", 20)
            state.update("@a", 5)
            state.save()
            self.assertEqual(list(json.loads(path.read_text(encoding="utf-8"))["channels"]), ["@a", "@b"])
            reloaded = SeenState(path)
            self.assertEqual(reloaded.get("@a"), 5)
            self.assertEqual(reloaded.get("@b"), 20)

    def test_corrupt_state_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("{not json", encoding="utf-8")
            state = SeenState(path)
            self.assertIsNone(state.get("@a"))
            path.write_text(json.dumps({"channels": {"@x": "y", "@z": 3}}), encoding="utf-8")
            reloaded = SeenState(path)
            self.assertIsNone(reloaded.get("@x"))
            self.assertEqual(reloaded.get("@z"), 3)


class NewsStoreTests(unittest.TestCase):
    def test_save_items_and_load_them_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = NewsStore(Path(tmp))
            path, count = store.save_items([make_item(1), make_item(2)])
            self.assertEqual(count, 2)
            self.assertTrue(path.is_file())
            self.assertEqual(store.latest_raw_path(), path)
            loaded = store.load_items()
            self.assertEqual([item.message_id for item in loaded], [1, 2])
            self.assertEqual(loaded[0].channel_title, "Desk")
            self.assertEqual(loaded[0].permalink, "https://t.me/desk/1")

    def test_load_items_accepts_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom.jsonl"
            append_jsonl(target, [make_item(9).to_dict()])
            store = NewsStore(Path(tmp) / "data")
            items = store.load_items(target)
            self.assertEqual([item.message_id for item in items], [9])

    def test_load_items_without_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = NewsStore(Path(tmp) / "data")
            self.assertEqual(store.load_items(), [])

    def test_known_uids_covers_recent_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = NewsStore(Path(tmp))
            store.save_items([make_item(1)])
            now = datetime.now(timezone.utc)
            self.assertIn("c:1", store.known_uids(days=3, now=now))
            self.assertNotIn("c:2", store.known_uids(days=3, now=now))
            far_future = now + timedelta(days=400)
            self.assertEqual(store.known_uids(days=1, now=far_future), set())

    def test_save_report_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = NewsStore(Path(tmp))
            report = AnalysisReport(
                provider="ollama",
                model="qwen2.5:3b",
                generated_at=datetime(2026, 6, 1, 12, 30, 45, tzinfo=timezone.utc),
                item_count=2,
                overall={"sentiment": "bearish"},
            )
            path = store.save_report(report, tag="demo")
            self.assertEqual(path.name, "analysis-2026_06_01_12_30-demo.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall"]["sentiment"], "bearish")
            self.assertEqual(payload["generated_at"], "2026-06-01T12:30:45+00:00")
            self.assertEqual(AnalysisReport.from_dict(payload).item_count, 2)

    def test_save_report_without_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = NewsStore(Path(tmp))
            path = store.save_report(AnalysisReport(provider="p", model="m"))
            self.assertTrue(path.name.startswith("analysis-"))
            self.assertFalse(path.name.endswith("-demo.json"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

