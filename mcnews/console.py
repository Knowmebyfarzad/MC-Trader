"""Tiny console helper so the CLI stays readable without extra dependencies.

Everything is ASCII on purpose so it prints correctly on a Windows console
(cp1252/cp437) as well as in redirected output.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}

_enabled: bool = False


def init_stdout() -> None:
    """Force UTF-8 stdout so non-latin channel text does not crash on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - depending on shell
                pass


def enable(flag: bool | None = None) -> bool:
    """Enable/disable ANSI colours. ``None`` = auto-detect."""
    global _enabled
    if flag is None:
        flag = bool(
            getattr(sys.stdout, "isatty", lambda: False)()
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM", "") != "dumb"
        )
    _enabled = bool(flag)
    return _enabled


def style(text: str, *styles: str) -> str:
    if not _enabled or not styles:
        return text
    prefix = "".join("\033[%sm" % _CODES.get(s, "0") for s in styles)
    return "%s%s\033[0m" % (prefix, text)


def info(message: str) -> None:
    print(message)


def ok(message: str) -> None:
    print("%s %s" % (style("[ OK ]", "green", "bold"), message))


def warn(message: str) -> None:
    print("%s %s" % (style("[WARN]", "yellow", "bold"), message))


def err(message: str) -> None:
    print("%s %s" % (style("[FAIL]", "red", "bold"), message))


def head(message: str) -> None:
    print(style(message, "cyan", "bold"))


def rule(char: str = "-", width: int = 78) -> None:
    print(style(char * width, "dim"))


def field(label: str, value: Any) -> None:
    print("  %s %s" % (style(label.ljust(18), "dim"), value))
