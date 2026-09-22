"""Tests for NewsItem / AnalysisReport (Telegram objects are duck-typed)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mcnews.models import AnalysisReport, NewsItem, permalink_for


class FakeMessage:
    """Minimal stand-in for a telethon Message (attributes set on demand)."""

    def __init__(self, message_id, text=None, date=None, media=None, views=None, forwards=None):
        self.id = message_id
        if text is not None:
            self.message = text
        self.date = date
        if media is not None:
            self.media = media
        if views is not None:
            self.views = views
        if forwards is not None:
            self.forwards = forwards


class FakeMedia:
    """Any non-None media object; only its class name is used."""

    def __init__(self, name="MessageMediaPhoto"):
        self.__class__.__name__ = name


class NewsItemFactoryTests(unittest.TestCase):
    def test_builds_from_message(self):
        message = FakeMessage(
            77,
            "Breaking: something happened",
            datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
            views=120,
        )
        item = NewsItem.from_message(message, channel="-100123", channel_title="Desk", channel_username="desk")
        self.assertEqual(item.message_id, 77)
        self.assertEqual(item.channel, "-100123")
        self.assertEqual(item.label, "Desk")
        self.assertEqual(item.channel_username, "desk")
        self.assertEqual(item.views, 120)
        self.assertIsNone(item.forwards)
        self.assertFalse(item.has_media)
        self.assertEqual(item.permalink, "https://t.me/desk/77")
        self.assertTrue(item.is_usable())

    def test_naive_datetime_becomes_utc(self):
        item = NewsItem.from_message(FakeMessage(1, "x", datetime(2026, 5, 1, 12, 0)), channel="c")
        self.assertEqual(item.date.tzinfo, timezone.utc)

    def test_missing_attributes_are_tolerated(self):
        item = NewsItem.from_message(FakeMessage(2), channel="c")
        self.assertEqual(item.text, "")
        self.assertFalse(item.is_usable())
        self.assertFalse(item.has_media)
        self.assertEqual(item.date.tzinfo, timezone.utc)

    def test_media_only_post_is_flagged_but_not_usable(self):
        item = NewsItem.from_message(FakeMessage(3, None, media=FakeMedia("MessageMediaDocument")), channel="c")
        self.assertTrue(item.has_media)
        self.assertEqual(item.media_type, "MessageMediaDocument")
        self.assertFalse(item.is_usable())

    def test_media_caption_is_used_as_text(self):
        item = NewsItem.from_message(FakeMessage(4, "caption text", media=FakeMedia()), channel="c")
        self.assertEqual(item.text, "caption text")
        self.assertTrue(item.is_usable())

    def test_label_falls_back_to_channel_key(self):
        item = NewsItem(channel="-1009", message_id=5, text="hello")
        self.assertEqual(item.label, "-1009")

    def test_username_is_normalised(self):
        item = NewsItem(channel="-1009", message_id=5, text="hello", channel_username="@desk")
        self.assertEqual(item.channel_username, "desk")
        self.assertEqual(item.permalink, "https://t.me/desk/5")

    def test_uid_is_channel_plus_message_id(self):
        item = NewsItem(channel="-1009", message_id=5, text="hello")
        self.assertEqual(item.uid, "-1009:5")

    def test_label_line_contains_key_fields(self):
        item = NewsItem(
            channel="-1009",
            message_id=5,
            text="hello",
            date=datetime(2026, 3, 4, 5, 6, tzinfo=timezone.utc),
            channel_title="Desk",
            channel_username="desk",
        )
        line = item.label_line()
        self.assertIn("Desk", line)
        self.assertIn("#5", line)
        self.assertIn("2026-03-04 05:06 UTC", line)
        self.assertIn("https://t.me/desk/5", line)


class NewsItemSerialisationTests(unittest.TestCase):
    def test_roundtrip_preserves_fields(self):
        original = NewsItem(
            channel="-1009",
            message_id=5,
            text="hello",
            date=datetime(2026, 3, 4, 5, 6, tzinfo=timezone.utc),
            channel_title="Desk",
            channel_username="desk",
            has_media=True,
            media_type="MessageMediaPhoto",
            views=10,
            forwards=2,
        )
        revived = NewsItem.from_dict(original.to_dict())
        self.assertEqual(revived.uid, original.uid)
        self.assertEqual(revived.date, original.date)
        self.assertEqual(revived.views, 10)
        self.assertTrue(revived.has_media)
        self.assertEqual(revived.permalink, original.permalink)

    def test_unknown_fields_are_ignored(self):
        revived = NewsItem.from_dict({"channel": "c", "message_id": 1, "text": "t", "unknown": "x"})
        self.assertEqual(revived.message_id, 1)

    def test_broken_date_falls_back_to_now(self):
        revived = NewsItem.from_dict({"channel": "c", "message_id": 1, "text": "t", "date": "not-a-date"})
        self.assertEqual(revived.date.tzinfo, timezone.utc)


class ReportTests(unittest.TestCase):
    def test_roundtrip(self):
        report = AnalysisReport(
            provider="ollama",
            model="qwen2.5:3b",
            item_count=3,
            batch_count=1,
            parsed_batches=1,
            overall={"sentiment": "bullish"},
            items=[{"channel": "c", "message_id": 1, "impact": 5}],
            watchlist=["gold"],
        )
        revived = AnalysisReport.from_dict(report.to_dict())
        self.assertEqual(revived.overall, {"sentiment": "bullish"})
        self.assertEqual(revived.items[0]["impact"], 5)
        self.assertEqual(revived.generated_at.tzinfo, timezone.utc)

    def test_permalink_helper(self):
        self.assertIsNone(permalink_for(None, 5))
        self.assertEqual(permalink_for("@desk", 5), "https://t.me/desk/5")
        self.assertEqual(permalink_for("desk", 5), "https://t.me/desk/5")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

