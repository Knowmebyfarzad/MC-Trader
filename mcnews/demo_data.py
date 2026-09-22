"""Synthetic sample posts used by ``python main.py demo``.

These texts are FICTIONAL and exist only to prove that the local model half of
the pipeline works before Telegram credentials are available.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import NewsItem

SAMPLE_POSTS: list[dict[str, object]] = [
    {
        "channel": "-1009000000001",
        "channel_title": "Sample Macro Wire (demo)",
        "channel_username": "sample_macro",
        "message_id": 4821,
        "minutes_ago": 18,
        "text": (
            "Central bank of country X kept its policy rate unchanged at 6.5% today, "
            "matching expectations. The statement dropped the phrase 'additional "
            "tightening may be appropriate' and instead said the board will 'assess "
            "incoming data'. Two members dissented in favour of a 25bp cut."
        ),
    },
    {
        "channel": "-1009000000001",
        "channel_title": "Sample Macro Wire (demo)",
        "channel_username": "sample_macro",
        "message_id": 4822,
        "minutes_ago": 12,
        "text": (
            "Follow-up: the finance ministry published its monthly budget statement, "
            "showing a deficit of 1.8% of GDP year-to-date versus 2.3% a year earlier. "
            "Tax revenue rose 11% y/y. No new bond issuance was announced."
        ),
    },
    {
        "channel": "-1009000000002",
        "channel_title": "Sample Metals Desk (demo)",
        "channel_username": "sample_metals",
        "message_id": 990,
        "minutes_ago": 40,
        "text": (
            "Unconfirmed: a large smelter in region Y reportedly halted one of three "
            "production lines after an equipment failure. The company has not "
            "confirmed this. Local traders report a 0.4% move in the physical "
            "premium but volumes are very thin."
        ),
    },
    {
        "channel": "-1009000000003",
        "channel_title": "Sample Crypto Alert (demo)",
        "channel_username": "sample_crypto",
        "message_id": 77,
        "minutes_ago": 5,
        "text": (
            "JUST IN: exchange Z says withdrawals are temporarily paused for "
            "maintenance. The team says customer funds are safe and expects service "
            "to resume within 2 hours. This is the second pause in 30 days."
        ),
    },
    {
        "channel": "-1009000000004",
        "channel_title": "Sample Promo Channel (demo)",
        "channel_username": "sample_promo",
        "message_id": 31337,
        "minutes_ago": 3,
        "text": (
            "LAST CHANCE!!! Join our VIP signals group now, 90% win rate, 10x profit "
            "guaranteed!! Send a DM to claim your spot before it closes tonight!!!"
        ),
    },
]


def build_sample_items(now: datetime | None = None) -> list[NewsItem]:
    """Turn :data:`SAMPLE_POSTS` into ``NewsItem`` objects."""
    reference = now or datetime.now(timezone.utc)
    items: list[NewsItem] = []
    for post in SAMPLE_POSTS:
        items.append(
            NewsItem(
                channel=str(post["channel"]),
                channel_title=str(post["channel_title"]),
                channel_username=str(post.get("channel_username") or "") or None,
                message_id=int(post["message_id"]),
                text=str(post["text"]),
                date=reference - timedelta(minutes=int(post["minutes_ago"])),
                has_media=False,
            )
        )
    return items
