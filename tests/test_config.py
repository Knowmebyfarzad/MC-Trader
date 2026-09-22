"""Tests for .env parsing and Settings resolution (no network access)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcnews.config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OPENAI_URL,
    Settings,
    load_env_file,
    parse_env_line,
    split_list,
)

CLEAN_ENV = {
    key: value
    for key, value in os.environ.items()
    if not key.startswith(("TELEGRAM_", "LLM_", "DATA_DIR", "MAX_MESSAGE", "BATCH_", "FLOOD_"))
}


class ParseEnvLineTests(unittest.TestCase):
    def test_plain_pair(self):
        self.assertEqual(parse_env_line("KEY=value"), ("KEY", "value"))

    def test_comments_and_blanks_are_ignored(self):
        self.assertIsNone(parse_env_line(""))
        self.assertIsNone(parse_env_line("   "))
        self.assertIsNone(parse_env_line("# comment"))
        self.assertIsNone(parse_env_line("no equals sign here"))

    def test_export_prefix_and_quotes(self):
        self.assertEqual(parse_env_line("export A=1"), ("A", "1"))
        self.assertEqual(parse_env_line('B="x y"'), ("B", "x y"))
        self.assertEqual(parse_env_line("C='y z'"), ("C", "y z"))

    def test_inline_comment_is_dropped_for_unquoted_values(self):
        self.assertEqual(parse_env_line("D=1 # trailing"), ("D", "1"))
        self.assertEqual(parse_env_line('E="1 # kept"'), ("E", "1 # kept"))

    def test_value_may_contain_equals_sign(self):
        self.assertEqual(parse_env_line("F=a=b"), ("F", "a=b"))


class SplitListTests(unittest.TestCase):
    def test_splits_all_separators(self):
        self.assertEqual(split_list("a, b;;c\nd"), ["a", "b", "c", "d"])

    def test_none_and_empty(self):
        self.assertEqual(split_list(None), [])
        self.assertEqual(split_list(""), [])
        self.assertEqual(split_list(",,"), [])

    def test_iterable_input(self):
        self.assertEqual(split_list(["@a,@b", "@c"]), ["@a", "@b", "@c"])


class LoadEnvFileTests(unittest.TestCase):
    def test_reads_file_and_sets_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nTELEGRAM_API_ID=12345\nTELEGRAM_API_HASH=abcdef\nTELEGRAM_CHANNELS=@a, @b\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
                parsed = load_env_file(path)
                self.assertEqual(parsed["TELEGRAM_API_ID"], "12345")
                self.assertEqual(os.environ["TELEGRAM_CHANNELS"], "@a, @b")

    def test_existing_environment_wins_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"LLM_MODEL": "from-env"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["LLM_MODEL"], "from-env")
                load_env_file(path, override=True)
                self.assertEqual(os.environ["LLM_MODEL"], "from-file")

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(load_env_file(Path("does-not-exist.env")), {})


class SettingsTests(unittest.TestCase):
    def test_defaults_without_env(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.resolved_base_url(), DEFAULT_OLLAMA_URL)
        self.assertEqual(settings.channels, [])
        self.assertFalse(settings.session_exists())

    def test_openai_provider_uses_lm_studio_default_url(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.resolved_base_url(), DEFAULT_OPENAI_URL)

    def test_unknown_provider_falls_back_to_ollama(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "banana"}, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.provider, "ollama")

    def test_base_url_override_is_normalised(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:8080/v1/"}, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.resolved_base_url(), "http://127.0.0.1:8080/v1")

    def test_values_are_read_from_env(self):
        env = dict(CLEAN_ENV)
        env.update(
            {
                "TELEGRAM_API_ID": "987",
                "TELEGRAM_API_HASH": "deadbeef",
                "TELEGRAM_PHONE": "+10000000000",
                "TELEGRAM_CHANNELS": "@a, https://t.me/b",
                "LLM_MODEL": "qwen3:4b",
                "LLM_TEMPERATURE": "0.7",
                "LLM_TIMEOUT": "30",
                "BATCH_CHARS": "1234",
            }
        )
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.api_id, 987)
        self.assertEqual(settings.api_hash, "deadbeef")
        self.assertEqual(settings.phone, "+10000000000")
        self.assertEqual(settings.channels, ["@a", "https://t.me/b"])
        self.assertEqual(settings.model, "qwen3:4b")
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.timeout, 30)
        self.assertEqual(settings.batch_chars, 1234)

    def test_telegram_problems(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            settings = Settings.from_env(None)
            self.assertEqual(len(settings.telegram_problems()), 2)
            settings.api_id = 1
            settings.api_hash = "x"
            self.assertEqual(settings.telegram_problems(), [])

    def test_relative_paths_are_resolved_from_project_root(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_SESSION": "sessions/test_sess"}, clear=True):
            settings = Settings.from_env(None)
        self.assertTrue(settings.session_path.is_absolute())
        self.assertEqual(settings.session_path.name, "test_sess")

    def test_data_dir_is_read_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"DATA_DIR": tmp}, clear=True):
                settings = Settings.from_env(None)
            self.assertEqual(settings.data_dir, Path(tmp))

    def test_relative_data_dir_is_resolved_from_project_root(self):
        with mock.patch.dict(os.environ, {"DATA_DIR": "my_data"}, clear=True):
            settings = Settings.from_env(None)
        self.assertTrue(settings.data_dir.is_absolute())
        self.assertEqual(settings.data_dir.name, "my_data")

    def test_data_dir_defaults_to_project_data(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.data_dir.name, "data")

    def test_ensure_dirs_creates_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp) / "data", session_path=Path(tmp) / "sess" / "s")
            settings.ensure_dirs()
            self.assertTrue((Path(tmp) / "data" / "raw").is_dir())
            self.assertTrue((Path(tmp) / "data" / "analyses").is_dir())
            self.assertTrue((Path(tmp) / "sess").is_dir())

    def test_session_exists_detects_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "mcnews"
            settings = Settings(session_path=session_path)
            self.assertFalse(settings.session_exists())
            session_path.with_suffix(".session").write_text("", encoding="utf-8")
            self.assertTrue(settings.session_exists())

    def test_describe_lists_missing_credentials(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True):
            described = dict(Settings.from_env(None).describe())
        self.assertEqual(described["Telegram api_id"], "<missing>")
        self.assertIn("LLM model", described)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

