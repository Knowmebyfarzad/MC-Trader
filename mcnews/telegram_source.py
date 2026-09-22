"""Read-only Telegram access via Telethon.

The reader never writes to Telegram: it only resolves entities, iterates
messages and lists dialogs. Channel references may be written as ``@name``,
``https://t.me/name``, ``t.me/s/name``, ``t.me/c/<internal_id>/<msg>`` or a raw
numeric id.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import Settings
from .models import NewsItem
from .store import SeenState

logger = logging.getLogger(__name__)

try:  # pragma: no cover - telethon is a real dependency; the fallback keeps the
    # module importable (and unit-testable) on machines without it.
    from telethon.errors import (
        AuthKeyUnregisteredError,
        ChannelPrivateError,
        ChatAdminRequiredError,
        FloodWaitError,
        RPCError,
        UsernameInvalidError,
        UsernameNotOccupiedError,
    )
    from telethon.tl.types import Channel, Chat, User

    TELEGRAM_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    TELEGRAM_IMPORT_ERROR = exc

    class _MissingTelethonError(Exception):
        seconds = 0

    FloodWaitError = ChannelPrivateError = ChatAdminRequiredError = RPCError = _MissingTelethonError
    UsernameInvalidError = UsernameNotOccupiedError = AuthKeyUnregisteredError = _MissingTelethonError
    Channel = Chat = User = ()  # type: ignore[assignment]

LINK_RE = re.compile(r"^(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me|telegram\.dog)/(?P<rest>.+)$", re.I)
NUMERIC_RE = re.compile(r"^-?\d+$")
USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,31}$")


class TelegramError(RuntimeError):
    """Generic, user-facing Telegram problem."""


class TelegramNotConfigured(TelegramError):
    """Missing credentials or missing telethon."""


class TelegramAuthRequired(TelegramError):
    """A session exists but is not logged in yet."""


# ------------------------------------------------------------------- references
def normalize_channel_ref(raw: str) -> str:
    """Turn any common channel reference into ``@username``/numeric id/invite."""
    token = (raw or "").strip().strip('"').strip("'").rstrip("/")
    if not token:
        raise ValueError("empty channel reference")

    match = LINK_RE.match(token)
    if match:
        token = match.group("rest")
    token = token.lstrip("/")

    parts = [p for p in token.split("/") if p]
    if parts and parts[0].lower() == "s" and len(parts) > 1:
        parts = parts[1:]  # t.me/s/name preview links

    if parts and parts[0].lower() == "c":
        if len(parts) >= 2 and parts[1].isdigit():
            return "-100%s" % parts[1]
        raise ValueError("cannot parse private channel link (expected t.me/c/<id>/<msg>)")

    if not parts:
        raise ValueError("empty channel reference")

    if parts[0].lower() == "joinchat":
        if len(parts) >= 2 and parts[1]:
            return "+" + parts[1]
        raise ValueError("incomplete invite link (expected t.me/joinchat/<hash>)")
    if token.startswith("+"):
        return token

    head = parts[0]
    if head.startswith("@"):
        head = head[1:]
    if NUMERIC_RE.match(head):
        return head
    if not USERNAME_RE.match(head):
        raise ValueError("'%s' is not a valid @username or numeric id" % parts[0])
    return "@" + head


def parse_channel_refs(values: Iterable[str]) -> tuple[list[str], list[str]]:
    """Normalise a list of refs. Returns ``(refs, errors)`` preserving order."""
    refs: list[str] = []
    errors: list[str] = []
    for raw in values:
        if not raw or not str(raw).strip():
            continue
        try:
            ref = normalize_channel_ref(str(raw))
        except ValueError as exc:
            errors.append("%s (%s)" % (str(raw).strip(), exc))
            continue
        if ref not in refs:
            refs.append(ref)
    return refs, errors


def entity_kind(entity: Any) -> str:
    """Classify a Telethon entity into channel/group/bot/user/unknown.

    The ``isinstance`` checks are the fast path; the attribute based fallback
    keeps the helper correct for telegram-like objects (mocks, other telethon
    versions where the class identity differs).
    """
    if isinstance(entity, Channel):
        return "channel" if getattr(entity, "broadcast", False) else "group"
    if isinstance(entity, Chat):
        return "group"
    if isinstance(entity, User):
        return "bot" if getattr(entity, "bot", False) else "user"

    broadcast = getattr(entity, "broadcast", None)
    megagroup = getattr(entity, "megagroup", None)
    if broadcast is not None or megagroup is not None:
        return "channel" if broadcast else "group"
    if getattr(entity, "bot", None) is not None or getattr(entity, "first_name", None) is not None:
        return "bot" if getattr(entity, "bot", False) else "user"

    name = type(entity).__name__.lower()
    if "channel" in name:
        return "channel"
    if "group" in name or "chat" in name or "megagroup" in name:
        return "group"
    return "unknown"


def entity_meta(entity: Any, fallback_ref: str = "") -> dict[str, str]:
    """Stable key + display fields for an entity."""
    username = getattr(entity, "username", None) or ""
    title = getattr(entity, "title", None) or ""
    if not title:
        first = getattr(entity, "first_name", None) or ""
        last = getattr(entity, "last_name", None) or ""
        title = (first + " " + last).strip() or username
    identifier = getattr(entity, "id", None)
    channel = str(identifier) if identifier is not None else (fallback_ref or "unknown")
    return {
        "channel": channel,
        "title": title or channel,
        "username": str(username).lstrip("@"),
        "kind": entity_kind(entity),
    }


def matches_keyword(text: str, keywords: Sequence[str]) -> bool:
    """Case-insensitive OR match against cleaned keywords."""
    if not keywords:
        return True
    haystack = (text or "").lower()
    return any(word.lower() in haystack for word in keywords if word)


@dataclass
class DialogInfo:
    """One entry of ``GetDialogs`` (used by the ``channels`` command)."""

    ref: str
    id: str
    title: str
    username: str
    kind: str
    unread: int = 0

    def describe(self) -> str:
        unread = " unread=%d" % self.unread if self.unread else ""
        return "%-8s %-30s %sid=%s" % (self.kind, self.title[:30], unread, self.id)


@dataclass
class ChannelRead:
    """Outcome of reading one channel (per-channel diagnostics)."""

    ref: str
    ok: bool = False
    title: str = ""
    kind: str = ""
    channel: str = ""
    fetched: int = 0
    kept: int = 0
    skipped: int = 0
    filtered: int = 0
    error: str | None = None
    last_message_id: int | None = None
    items: list[NewsItem] = field(default_factory=list)

    def describe(self) -> str:
        if not self.ok:
            return "%-24s ERROR: %s" % (self.ref, self.error or "unknown error")
        return "%-24s %-8s read=%-4d kept=%-4d skipped=%-3d filtered=%-3d latest_id=%s" % (
            self.ref,
            self.kind or "?",
            self.fetched,
            self.kept,
            self.skipped,
            self.filtered,
            self.last_message_id if self.last_message_id is not None else "-",
        )


class TelegramNewsReader:
    """Thin, testable wrapper around ``telethon.TelegramClient``.

    ``client`` may be injected (used by the test suite) - in that case the
    reader neither builds nor disconnects it.
    """

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        self.client = client
        self._owns_client = client is None
        self._entities: dict[str, Any] = {}
        self._entity_errors: dict[str, str] = {}
        self._me: dict[str, Any] | None = None

    # ------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> "TelegramNewsReader":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def _build_client(self) -> Any:
        if TELEGRAM_IMPORT_ERROR is not None:
            raise TelegramNotConfigured("telethon is missing - pip install -r requirements.txt")
        from telethon import TelegramClient

        return TelegramClient(
            str(self.settings.session_path),
            int(self.settings.api_id or 0),
            str(self.settings.api_hash or ""),
            flood_sleep_threshold=int(self.settings.flood_sleep),
        )

    async def connect(self, interactive: bool = False) -> bool:
        """Connect and (optionally) log in. Raises on missing credentials."""
        problems = self.settings.telegram_problems()
        if problems:
            raise TelegramNotConfigured("; ".join(problems))
        if self.client is None:
            self.settings.ensure_dirs()
            self.client = self._build_client()
        if not self.client.is_connected():
            await self.client.connect()
        if not await self.client.is_user_authorized():
            if not interactive:
                raise TelegramAuthRequired(
                    "this session is not logged in yet - run:  python main.py login"
                )
            # Telethon prompts for the phone, the login code and the 2FA password
            # through its default callables, so the phone is only passed when known.
            if self.settings.phone:
                await self.client.start(phone=self.settings.phone)
            else:
                await self.client.start()
        return True

    async def close(self) -> None:
        if self.client is not None and self._owns_client:
            try:
                await self.client.disconnect()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("disconnect failed: %s", exc)
        if self._owns_client:
            self.client = None

    async def me(self) -> dict[str, Any]:
        """Account currently logged in (handy to prove the session works)."""
        if self._me is None:
            user = await self.client.get_me()
            first = getattr(user, "first_name", "") or ""
            last = getattr(user, "last_name", "") or ""
            self._me = {
                "id": getattr(user, "id", None),
                "username": getattr(user, "username", "") or "",
                "name": (first + " " + last).strip() or getattr(user, "username", "") or "",
                "phone": getattr(user, "phone", "") or "",
            }
        return self._me

    # --------------------------------------------------------------- resolving
    async def resolve(self, ref: str) -> Any:
        """Resolve a reference to a Telethon entity, caching results per run."""
        if ref in self._entities:
            return self._entities[ref]
        if ref in self._entity_errors:
            raise TelegramError(self._entity_errors[ref])

        message: str
        try:
            entity = await self.client.get_entity(ref)
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            message = "Telegram flood-wait of %ss (lower --limit or retry later)" % seconds
        except (UsernameNotOccupiedError, UsernameInvalidError):
            message = "%s does not exist (username not found)" % ref
        except ChannelPrivateError:
            message = "%s is private: this account is not a member (join it first)" % ref
        except ValueError as exc:
            message = (
                "cannot resolve %s (%s). Use an @username, or the numeric id of a "
                "channel this account already joined." % (ref, exc)
            )
        except AuthKeyUnregisteredError as exc:
            message = (
                "session revoked by Telegram (%s) - delete the .session file and run 'login' again"
                % exc
            )
        except RPCError as exc:
            message = "Telegram error while resolving %s: %s" % (ref, exc)
        except Exception as exc:  # unexpected: report instead of crashing the run
            message = "unexpected error while resolving %s: %s: %s" % (ref, type(exc).__name__, exc)
        else:
            self._entities[ref] = entity
            return entity

        self._entity_errors[ref] = message
        raise TelegramError(message)

    # ---------------------------------------------------------------- reading
    async def fetch_channel(
        self,
        ref: str,
        limit: int = 20,
        min_id: int | None = None,
        keywords: Sequence[str] | None = None,
        state: SeenState | None = None,
        only_new: bool = True,
    ) -> ChannelRead:
        """Read the newest posts of one channel. Never sends anything."""
        result = ChannelRead(ref=ref)
        try:
            entity = await self.resolve(ref)
        except TelegramError as exc:
            result.error = str(exc)
            return result

        meta = entity_meta(entity, ref)
        result.title = meta["title"]
        result.kind = meta["kind"]
        result.channel = meta["channel"]

        if min_id is None and state is not None and only_new:
            min_id = state.get(meta["channel"])

        request: dict[str, Any] = {"limit": max(1, int(limit))}
        if min_id:
            request["min_id"] = int(min_id)
        request["wait_time"] = None  # never block for minutes inside a batch read

        raw_messages: list[Any] = []
        try:
            async for message in self.client.iter_messages(entity, **request):
                raw_messages.append(message)
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            result.error = "flood-wait %ss: lower --limit or retry later" % seconds
            return result
        except ChannelPrivateError:
            result.error = "channel is private / no longer accessible for this account"
            return result
        except ChatAdminRequiredError:
            result.error = "reading this chat requires admin rights"
            return result
        except RPCError as exc:
            result.error = "Telegram error: %s" % exc
            return result

        result.fetched = len(raw_messages)
        for message in raw_messages:
            item = NewsItem.from_message(
                message,
                channel=meta["channel"],
                channel_title=meta["title"],
                channel_username=meta["username"] or None,
            )
            if result.last_message_id is None or item.message_id > result.last_message_id:
                result.last_message_id = item.message_id
            if not item.is_usable():
                result.skipped += 1
                continue
            if not matches_keyword(item.text, list(keywords or [])):
                result.filtered += 1
                continue
            result.items.append(item)

        result.kept = len(result.items)
        result.ok = True
        return result

    async def fetch_many(
        self,
        refs: Sequence[str],
        limit: int = 20,
        keywords: Sequence[str] | None = None,
        state: SeenState | None = None,
        only_new: bool = True,
    ) -> list[ChannelRead]:
        """Read several channels sequentially (politeness > speed)."""
        reads: list[ChannelRead] = []
        for ref in refs:
            read = await self.fetch_channel(
                ref, limit=limit, keywords=keywords, state=state, only_new=only_new
            )
            reads.append(read)
            if read.ok and state is not None:
                state.update(read.channel or read.ref, read.last_message_id)
            elif not read.ok:
                logger.warning("channel %s failed: %s", ref, read.error)
            await asyncio.sleep(0.4)  # small pause to stay friendly with the API
        return reads

    async def list_dialogs(self, limit: int = 300) -> list[DialogInfo]:
        """List channels/groups this account can read (build your --channels)."""
        dialogs: list[DialogInfo] = []
        async for dialog in self.client.iter_dialogs(limit=max(1, int(limit))):
            entity = getattr(dialog, "entity", None)
            meta = entity_meta(entity, str(getattr(dialog, "id", "")))
            kind = meta["kind"]
            if kind in ("user", "bot"):
                continue
            ref = "@%s" % meta["username"] if meta["username"] else meta["channel"]
            dialogs.append(
                DialogInfo(
                    ref=ref,
                    id=meta["channel"],
                    title=meta["title"],
                    username=meta["username"],
                    kind=kind,
                    unread=int(getattr(dialog, "unread_count", 0) or 0),
                )
            )
        dialogs.sort(key=lambda d: (d.kind, d.title.lower()))
        return dialogs


def ref_to_link(ref: str) -> str:
    """Public link for a reference when one exists (numeric ids have none)."""
    if ref.startswith("@"):
        return "https://t.me/%s" % ref[1:]
    return ref
