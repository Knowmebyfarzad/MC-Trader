"""Configuration: a tiny .env loader plus a typed Settings object.

No external dependency is used on purpose (``python-dotenv`` is nice but this
MVP only needs KEY=VALUE lines).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen2.5:3b"

SUPPORTED_PROVIDERS = ("ollama", "openai")


def parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one ``KEY=VALUE`` line. Returns ``None`` for blanks/comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key, _, raw = stripped.partition("=")
    key = key.strip()
    if not key:
        return None
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        hash_index = value.find(" #")
        if hash_index != -1:
            value = value[:hash_index].rstrip()
    return key, value


def load_env_file(path: str | Path = DEFAULT_ENV_FILE, override: bool = False) -> dict[str, str]:
    """Load a .env file into ``os.environ`` and return the parsed mapping."""
    env_path = Path(path)
    parsed: dict[str, str] = {}
    if not env_path.is_file():
        return parsed
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return parsed
    for line in content.splitlines():
        pair = parse_env_line(line)
        if pair is None:
            continue
        key, value = pair
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def split_list(raw: str | Iterable[str] | None) -> list[str]:
    """Split comma/semicolon/newline separated values into a clean list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.replace(";", ",").replace("\n", ",").split(",")
    else:
        parts = []
        for chunk in raw:
            parts.extend(str(chunk).replace(";", ",").replace("\n", ",").split(","))
    return [p.strip() for p in parts if p and p.strip()]


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings resolved from .env / environment variables."""

    # --- Telegram (read only) -------------------------------------------------
    api_id: int | None = None
    api_hash: str | None = None
    phone: str | None = None
    session_path: Path = PROJECT_ROOT / "sessions" / "mcnews"
    channels: list[str] = field(default_factory=list)

    # --- Local LLM -----------------------------------------------------------
    provider: str = "ollama"
    base_url: str | None = None
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    temperature: float = 0.2
    timeout: int = 240
    num_ctx: int = 8192
    max_tokens: int = 1024

    # --- Pipeline ------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    max_message_chars: int = 1500
    batch_chars: int = 6000
    flood_sleep: int = 60

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_env(cls, env_file: str | Path | None = DEFAULT_ENV_FILE) -> "Settings":
        if env_file is not None:
            load_env_file(env_file)

        provider = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            provider = "ollama"

        session_raw = (os.environ.get("TELEGRAM_SESSION") or "").strip()
        data_raw = (os.environ.get("DATA_DIR") or "").strip()

        session_path = Path(session_raw) if session_raw else PROJECT_ROOT / "sessions" / "mcnews"
        if not session_path.is_absolute():
            session_path = (PROJECT_ROOT / session_path).resolve()

        data_dir = Path(data_raw) if data_raw else PROJECT_ROOT / "data"
        if not data_dir.is_absolute():
            data_dir = (PROJECT_ROOT / data_dir).resolve()

        return cls(
            api_id=_env_int("TELEGRAM_API_ID"),
            api_hash=(os.environ.get("TELEGRAM_API_HASH") or "").strip() or None,
            phone=(os.environ.get("TELEGRAM_PHONE") or "").strip() or None,
            session_path=session_path,
            channels=split_list(os.environ.get("TELEGRAM_CHANNELS")),
            provider=provider,
            base_url=(os.environ.get("LLM_BASE_URL") or "").strip() or None,
            model=(os.environ.get("LLM_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            api_key=(os.environ.get("LLM_API_KEY") or "").strip() or None,
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            timeout=_env_int("LLM_TIMEOUT", 240) or 240,
            num_ctx=_env_int("LLM_NUM_CTX", 8192) or 8192,
            max_tokens=_env_int("LLM_MAX_TOKENS", 1024) or 1024,
            max_message_chars=_env_int("MAX_MESSAGE_CHARS", 1500) or 1500,
            batch_chars=_env_int("BATCH_CHARS", 6000) or 6000,
            flood_sleep=_env_int("FLOOD_SLEEP", 60) or 60,
            data_dir=data_dir,
        )

    # ---------------------------------------------------------------- derived
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return DEFAULT_OLLAMA_URL if self.provider == "ollama" else DEFAULT_OPENAI_URL

    def session_exists(self) -> bool:
        return self.session_path.with_suffix(".session").is_file()

    def telegram_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.api_id:
            problems.append("TELEGRAM_API_ID is not set (get it from https://my.telegram.org)")
        if not self.api_hash:
            problems.append("TELEGRAM_API_HASH is not set (get it from https://my.telegram.org)")
        return problems

    def ensure_dirs(self) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "analyses").mkdir(parents=True, exist_ok=True)

    def describe(self) -> list[tuple[str, str]]:
        return [
            ("Telegram api_id", str(self.api_id) if self.api_id else "<missing>"),
            ("Telegram phone", self.phone or "<unset - will be prompted>"),
            ("Session file", str(self.session_path.with_suffix(".session"))),
            ("Channels (.env)", ", ".join(self.channels) if self.channels else "<none>"),
            ("LLM provider", self.provider),
            ("LLM base url", self.resolved_base_url()),
            ("LLM model", self.model),
            ("Data dir", str(self.data_dir)),
        ]
