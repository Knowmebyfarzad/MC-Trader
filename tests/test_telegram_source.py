"""Tests for the read-only Telegram layer.

The Telethon client is replaced by a fake, so no credentials or network access
are needed. The fake mimics the small surface the reader actually uses:
``is_connected``, ``connect``, ``is_user_authorized``, ``get_me``,
``get_entity``, ``iter_messages`` and ``iter_dialogs``.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from mcnews.config import Settings
from mcnews.store import SeenState
from mcnews.telegram_source import (
    ChannelRead,
    TelegramAuthRequired,
    TelegramError,
    TelegramNewsReader,
    TelegramNotConfigured,
    entity_kind,
    entity_meta,
    matches_keyword,
    normalize_channel_ref,
    parse_channel_refs,
    ref_to_link,
)

import tempfile
from pathlib import Path


class RefNormalisationTests(unittest.TestCase):
    def test_username_variants(self):
        for raw in ("@desk", "desk", "t.me/desk", "https://t.me/desk", "https://t.me/s/desk", "telegram.me/desk/123"):
            self.assertEqual(normalize_channel_ref(raw), "@desk", raw)

    def test_numeric_ids_are_kept(self):
        self.assertEqual(normalize_channel_ref("-1001234567890"), "-1001234567890")
        self.assertEqual(normalize_channel_ref("123456789"), "123456789")

    def test_private_link_becomes_prefixed_id(self):
        self.assertEqual(normalize_channel_ref("https://t.me/c/1234567890/42"), "-1001234567890")

    def test_invite_links_are_preserved(self):
        self.assertEqual(normalize_channel_ref("https://t.me/joinchat/AAAAEE"), "+AAAAEE")
        self.assertEqual(normalize_channel_ref("https://t.me/+AAAAEE"), "+AAAAEE")

    def test_trailing_slashes_and_quotes_are_stripped(self):
        self.assertEqual(normalize_channel_ref('"@desk/"'), "@desk")

    def test_invalid_values_raise(self):
        for raw in ("", "   ", "t.me/ab", "t.me/c/notanumber", "@@@", "https://t.me/joinchat"):
            with self.assertRaises(ValueError, msg=raw):
                normalize_channel_ref(raw)

    def test_parse_refs_dedupes_and_collects_errors(self):
        refs, errors = parse_channel_refs(["@news_desk", "https://t.me/news_desk", "bad-name!", "", "@gold_wire"])
        self.assertEqual(refs, ["@news_desk", "@gold_wire"])
        self.assertEqual(len(errors), 1)
        self.assertIn("bad-name!", errors[0])

    def test_ref_to_link(self):
        self.assertEqual(ref_to_link("@desk"), "https://t.me/desk")
        self.assertEqual(ref_to_link("-100123"), "-100123")
        self.assertEqual(ref_to_link("+invite"), "+invite")


class FakeChannelInfo:
    def __init__(self, identifier=123456, title="Desk", username="desk", broadcast=True, megagroup=False):
        self.id = identifier
        self.title = title
        self.username = username
        self.broadcast = broadcast
        self.megagroup = megagroup


class FakeUserInfo:
    def __init__(self, identifier=7, first_name="Ann", username="ann", bot=False):
        self.id = identifier
        self.first_name = first_name
        self.username = username
        self.bot = bot


class EntityTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(entity_kind(FakeChannelInfo()), "channel")
        self.assertEqual(entity_kind(FakeChannelInfo(broadcast=False, megagroup=True)), "group")
        self.assertEqual(entity_kind(FakeUserInfo()), "user")
        self.assertEqual(entity_kind(FakeUserInfo(bot=True)), "bot")
        self.assertEqual(entity_kind(object()), "unknown")

    def test_meta_prefers_numeric_id_as_stable_key(self):
        meta = entity_meta(FakeChannelInfo(), "@fallback")
        self.assertEqual(meta["channel"], "123456")
        self.assertEqual(meta["title"], "Desk")
        self.assertEqual(meta["username"], "desk")
        self.assertEqual(meta["kind"], "channel")

    def test_meta_uses_name_for_users(self):
        meta = entity_meta(FakeUserInfo(), "7")
        self.assertEqual(meta["title"], "Ann")
        self.assertEqual(meta["kind"], "user")

    def test_keyword_matching(self):
        self.assertTrue(matches_keyword("Gold is up", []))
        self.assertTrue(matches_keyword("Gold is up", ["gold"]))
        self.assertTrue(matches_keyword("GOLD IS UP", ["gold", "usd"]))
        self.assertFalse(matches_keyword("silver is up", ["gold"]))


try:
    from telethon.errors import FloodWaitError

    FLOOD_WAIT = FloodWaitError(request=None, capture=42)
except Exception:  # pragma: no cover - only if telethon is absent
    FLOOD_WAIT = None


class AsyncIter:
    """Minimal async iterator (telethon's iter_* return async iterators)."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class FakeMessage:
    def __init__(self, message_id, text=None, minutes_ago=0, media=None):
        self.id = message_id
        if text is not None:
            self.message = text
        self.date = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        if media is not None:
            self.media = media


class FakeDialog:
    def __init__(self, entity, unread_count=0):
        self.entity = entity
        self.id = entity.id
        self.unread_count = unread_count


class FakeTelethonClient:
    """Records every call so the reader's behaviour can be asserted."""

    def __init__(self, entities=None, messages=None, dialogs=None, authorized=True, errors=None):
        self.entities = dict(entities or {})
        self.messages = dict(messages or {})
        self.dialogs = list(dialogs or [])
        self.authorized = authorized
        self.errors = dict(errors or {})
        self.connected = False
        self.entity_calls: list[str] = []
        self.message_calls: list[dict] = []
        self.start_calls = 0

    def is_connected(self):
        return self.connected

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def is_user_authorized(self):
        return self.authorized

    async def start(self, *args, **kwargs):
        self.start_calls += 1
        self.authorized = True
        return self

    async def get_me(self):
        return FakeUserInfo()

    async def get_entity(self, ref):
        self.entity_calls.append(ref)
        if ref in self.errors:
            raise self.errors[ref]
        if ref not in self.entities:
            raise ValueError("Cannot find any entity corresponding to %r" % ref)
        return self.entities[ref]

    def iter_messages(self, entity, limit=None, min_id=0, wait_time=None, **kwargs):
        self.message_calls.append(
            {"entity_id": getattr(entity, "id", None), "limit": limit, "min_id": min_id, "wait_time": wait_time}
        )
        channel_id = getattr(entity, "id", None)
        found = sorted(self.messages.get(channel_id, []), key=lambda message: message.id, reverse=True)
        if min_id:
            found = [message for message in found if message.id > min_id]
        return AsyncIter(found[:limit] if limit else found)

    def iter_dialogs(self, limit=None):
        return AsyncIter(self.dialogs[:limit] if limit else self.dialogs)


class ReaderTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = Settings(
            api_id=1,
            api_hash="hash",
            session_path=self.root / "sessions" / "mcnews",
            data_dir=self.root / "data",
            flood_sleep=0,
        )
        self.channel = FakeChannelInfo(identifier=555, title="Desk", username="desk")

    def make_reader(self, messages=None, dialogs=None, authorized=True, errors=None):
        self.client = FakeTelethonClient(
            entities={"@desk": self.channel},
            messages={555: messages or []},
            dialogs=dialogs or [],
            authorized=authorized,
            errors=errors,
        )
        return TelegramNewsReader(self.settings, client=self.client)

    # ------------------------------------------------------------------ tests
    async def test_connect_uses_the_injected_client(self):
        reader = self.make_reader()
        self.assertTrue(await reader.connect())
        self.assertTrue(self.client.connected)
        self.assertEqual(self.client.start_calls, 0)

    async def test_missing_credentials_raise_not_configured(self):
        reader = TelegramNewsReader(Settings(session_path=self.root / "s"))
        with self.assertRaises(TelegramNotConfigured):
            await reader.connect()

    async def test_unauthorised_session_asks_for_login(self):
        reader = self.make_reader(authorized=False)
        with self.assertRaises(TelegramAuthRequired) as ctx:
            await reader.connect()
        self.assertIn("login", str(ctx.exception))
        self.assertEqual(self.client.start_calls, 0)

    async def test_interactive_login_starts_the_session(self):
        reader = self.make_reader(authorized=False)
        await reader.connect(interactive=True)
        self.assertEqual(self.client.start_calls, 1)

    async def test_me_returns_account_details(self):
        reader = self.make_reader()
        await reader.connect()
        account = await reader.me()
        self.assertEqual(account["name"], "Ann")
        self.assertEqual(account["username"], "ann")

    async def test_injected_client_is_not_disconnected(self):
        reader = self.make_reader()
        await reader.connect()
        await reader.close()
        self.assertTrue(self.client.connected)

    async def test_owned_client_is_disconnected_and_cleared(self):
        reader = TelegramNewsReader(self.settings)
        reader.client = self.client = FakeTelethonClient()
        self.client.connected = True
        await reader.close()
        self.assertFalse(self.client.connected)
        self.assertIsNone(reader.client)

    async def test_fetch_channel_counts_and_items(self):
        reader = self.make_reader(
            messages=[
                FakeMessage(3, "gold rally continues", minutes_ago=1),
                FakeMessage(2, "", minutes_ago=2),  # empty -> skipped
                FakeMessage(1, "ok text", minutes_ago=3),
            ]
        )
        await reader.connect()
        read = await reader.fetch_channel("@desk", limit=10)
        self.assertTrue(read.ok)
        self.assertEqual(read.fetched, 3)
        self.assertEqual(read.kept, 2)
        self.assertEqual(read.skipped, 1)
        self.assertEqual(read.filtered, 0)
        self.assertEqual(read.last_message_id, 3)
        self.assertEqual([item.message_id for item in read.items], [3, 1])
        self.assertEqual(read.items[0].channel, "555")
        self.assertEqual(read.items[0].channel_title, "Desk")
        self.assertEqual(read.items[0].permalink, "https://t.me/desk/3")
        self.assertEqual(read.kind, "channel")

    async def test_entity_is_resolved_only_once(self):
        reader = self.make_reader(messages=[FakeMessage(1, "text here")])
        await reader.connect()
        await reader.fetch_channel("@desk")
        await reader.fetch_channel("@desk")
        self.assertEqual(self.client.entity_calls, ["@desk"])

    async def test_keyword_filter(self):
        reader = self.make_reader(messages=[FakeMessage(2, "gold up"), FakeMessage(1, "silver up")])
        await reader.connect()
        read = await reader.fetch_channel("@desk", keywords=["gold"])
        self.assertEqual([item.message_id for item in read.items], [2])
        self.assertEqual(read.filtered, 1)

    async def test_state_supplies_min_id(self):
        reader = self.make_reader(messages=[FakeMessage(3, "new post"), FakeMessage(2, "old post")])
        await reader.connect()
        state = SeenState(self.root / "state.json")
        state.update("555", 2)
        read = await reader.fetch_channel("@desk", state=state)
        self.assertEqual(self.client.message_calls[0]["min_id"], 2)
        self.assertEqual([item.message_id for item in read.items], [3])

    async def test_state_is_ignored_when_only_new_is_false(self):
        reader = self.make_reader(messages=[FakeMessage(3, "new post")])
        await reader.connect()
        state = SeenState(self.root / "state.json")
        state.update("555", 2)
        await reader.fetch_channel("@desk", state=state, only_new=False)
        self.assertEqual(self.client.message_calls[0]["min_id"], 0)

    async def test_failed_resolution_is_reported(self):
        reader = self.make_reader(errors={"@missing": ValueError("Cannot find any entity")})
        await reader.connect()
        read = await reader.fetch_channel("@missing")
        self.assertFalse(read.ok)
        self.assertIn("cannot resolve", read.error or "")
        self.assertEqual(read.kept, 0)

    async def test_unexpected_entity_error_is_reported(self):
        reader = self.make_reader(errors={"@desk": RuntimeError("boom")})
        await reader.connect()
        read = await reader.fetch_channel("@desk")
        self.assertFalse(read.ok)
        self.assertIn("unexpected error while resolving", read.error or "")

    async def test_errors_are_cached_per_run(self):
        reader = self.make_reader(errors={"@desk": RuntimeError("boom")})
        await reader.connect()
        await reader.fetch_channel("@desk")
        await reader.fetch_channel("@desk")
        self.assertEqual(self.client.entity_calls, ["@desk"])

    @unittest.skipIf(FLOOD_WAIT is None, "telethon is not installed")
    async def test_flood_wait_is_reported(self):
        reader = self.make_reader(errors={"@desk": FLOOD_WAIT})
        await reader.connect()
        read = await reader.fetch_channel("@desk")
        self.assertFalse(read.ok)
        self.assertIn("42", read.error or "")

    async def test_fetch_many_updates_and_saves_state(self):
        reader = self.make_reader(messages=[FakeMessage(9, "latest news")])
        await reader.connect()
        state = SeenState(self.root / "state.json")
        reads = await reader.fetch_many(["@desk"], limit=5, state=state)
        self.assertEqual(len(reads), 1)
        self.assertEqual(state.get("555"), 9)
        state.save()
        self.assertTrue((self.root / "state.json").is_file())
        self.assertEqual(SeenState(self.root / "state.json").get("555"), 9)

    async def test_list_dialogs_skips_private_chats(self):
        reader = self.make_reader(
            dialogs=[
                FakeDialog(self.channel),
                FakeDialog(FakeUserInfo()),
                FakeDialog(
                    FakeChannelInfo(identifier=777, title="Group", username=None, broadcast=False, megagroup=True),
                    unread_count=4,
                ),
            ]
        )
        await reader.connect()
        dialogs = await reader.list_dialogs()
        self.assertEqual([dialog.kind for dialog in dialogs], ["channel", "group"])
        self.assertEqual(dialogs[0].ref, "@desk")
        self.assertEqual(dialogs[1].ref, "777")
        self.assertEqual(dialogs[1].unread, 4)

    def test_channel_read_describe(self):
        read = ChannelRead(ref="@desk")
        self.assertIn("ERROR", read.describe())
        read.ok = True
        read.kind = "channel"
        read.fetched, read.kept, read.last_message_id = 3, 2, 3
        line = read.describe()
        self.assertIn("read=3", line)
        self.assertIn("latest_id=3", line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()



