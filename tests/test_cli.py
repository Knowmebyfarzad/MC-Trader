"""Tests for the CLI argument layer and command dispatch (offline)."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mcnews.cli import EXIT_PROBLEM, UsageError, build_parser, main, resolve_channel_refs, resolve_settings
from mcnews.config import Settings

CLEAN_ENV = {
    key: value
    for key, value in os.environ.items()
    if not key.startswith(("TELEGRAM_", "LLM_", "DATA_DIR", "MAX_MESSAGE", "BATCH_", "FLOOD_"))
}


@contextlib.contextmanager
def quiet():
    """Swallow command output so the test report stays readable."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


class ParserTests(unittest.TestCase):
    def parse(self, argv):
        return build_parser().parse_args(argv)

    def test_every_command_parses(self):
        cases = [
            ["doctor"],
            ["login"],
            ["channels"],
            ["fetch", "-c", "@a_news"],
            ["analyze", "--from-raw"],
            ["watch", "--interval", "30", "--iterations", "2"],
            ["demo", "--model", "qwen3:4b"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertEqual(self.parse(argv).command, argv[0])

    def test_channels_are_repeatable(self):
        args = self.parse(["fetch", "-c", "@a_news,@b_news", "-c", "https://t.me/c_news"])
        self.assertEqual(args.channels, ["@a_news,@b_news", "https://t.me/c_news"])

    def test_dropped_channel_value_is_tolerated(self):
        """PowerShell drops a bare @word, so "-c -n 5" must not explode."""
        args = self.parse(["fetch", "-c", "-n", "5"])
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.channels, [""])

    def test_fetch_defaults(self):
        args = self.parse(["fetch"])
        self.assertEqual((args.limit, args.channels, args.show, args.json), (20, [], 10, False))

    def test_analyze_defaults(self):
        args = self.parse(["analyze"])
        self.assertIsNone(args.from_raw)
        self.assertFalse(args.no_save)
        self.assertIsNone(args.model)

    def test_watch_defaults(self):
        args = self.parse(["watch"])
        self.assertEqual((args.interval, args.iterations), (60, 0))

    def test_unknown_flag_exits_with_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            self.parse(["fetch", "--nope"])
        self.assertEqual(ctx.exception.code, 2)

    def test_global_flags_work_before_the_command(self):
        args = self.parse(["--env", "custom.env", "--verbose", "--no-color", "fetch"])
        self.assertEqual(args.env, "custom.env")
        self.assertTrue(args.verbose)
        self.assertTrue(args.no_color)

    def test_global_flags_work_after_the_command(self):
        args = self.parse(["fetch", "--env", "custom.env", "--verbose", "--no-color"])
        self.assertEqual(args.env, "custom.env")
        self.assertTrue(args.verbose)
        self.assertTrue(args.no_color)

    def test_global_flag_defaults(self):
        args = self.parse(["fetch"])
        self.assertIsNone(args.env)
        self.assertFalse(args.verbose)
        self.assertFalse(args.no_color)


class ResolveChannelRefsTests(unittest.TestCase):
    def make_args(self, *extra):
        return build_parser().parse_args(["fetch", *extra])

    def test_repeat_and_comma_separated_values_are_combined(self):
        settings = Settings(channels=["@from_env_news"])
        with quiet():
            refs = resolve_channel_refs(self.make_args("-c", "@a_news,@b_news", "-c", "@c_news"), settings)
        self.assertEqual(refs, ["@a_news", "@b_news", "@c_news", "@from_env_news"])

    def test_deduplicates_same_channel_written_twice(self):
        with quiet():
            refs = resolve_channel_refs(self.make_args("-c", "@a_news", "-c", "https://t.me/a_news"), Settings())
        self.assertEqual(refs, ["@a_news"])

    def test_invalid_reference_is_skipped_with_a_warning(self):
        with quiet() as buffer:
            refs = resolve_channel_refs(self.make_args("-c", "bad name!", "-c", "@ok_news"), Settings())
        self.assertEqual(refs, ["@ok_news"])
        self.assertIn("ignoring channel reference", buffer.getvalue())

    def test_empty_values_are_ignored(self):
        with self.assertRaises(UsageError):
            with quiet():
                resolve_channel_refs(self.make_args("-c", ""), Settings())

    def test_no_channels_mentions_powershell_quoting(self):
        with self.assertRaises(UsageError) as ctx:
            with quiet():
                resolve_channel_refs(self.make_args(), Settings())
        self.assertIn("PowerShell", str(ctx.exception))


class ResolveSettingsTests(unittest.TestCase):
    def test_cli_flags_override_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(CLEAN_ENV)
            env.update({"LLM_MODEL": "from-env", "DATA_DIR": tmp, "TELEGRAM_SESSION": str(Path(tmp) / "sess")})
            with mock.patch.dict(os.environ, env, clear=True):
                args = build_parser().parse_args(
                    [
                        "demo",
                        "--model",
                        "from-cli",
                        "--provider",
                        "openai",
                        "--base-url",
                        "http://x:1/v1/",
                        "--batch-chars",
                        "777",
                    ]
                )
                settings = resolve_settings(args)
        self.assertEqual(settings.model, "from-cli")
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.resolved_base_url(), "http://x:1/v1")
        self.assertEqual(settings.batch_chars, 777)

    def test_missing_env_file_is_a_usage_error(self):
        args = build_parser().parse_args(["--env", "nope-does-not-exist.env", "doctor"])
        with self.assertRaises(UsageError):
            with quiet():
                resolve_settings(args)


class DispatchTests(unittest.TestCase):
    def run_main(self, argv, extra_env=None):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(CLEAN_ENV)
            env.update({"DATA_DIR": tmp, "TELEGRAM_SESSION": str(Path(tmp) / "sess")})
            env.update(extra_env or {})
            with mock.patch.dict(os.environ, env, clear=True):
                with quiet():
                    return main(argv)

    def test_fetch_without_credentials_fails_cleanly(self):
        self.assertEqual(self.run_main(["fetch", "-c", "@x_news", "--no-color"]), EXIT_PROBLEM)

    def test_doctor_without_credentials_reports_problems(self):
        self.assertEqual(self.run_main(["doctor", "--no-color"]), EXIT_PROBLEM)

    def test_analyze_without_stored_posts_reports_problem(self):
        self.assertEqual(self.run_main(["analyze", "--from-raw", "--no-color"]), EXIT_PROBLEM)

    def test_dropped_channel_argument_reports_usage_error(self):
        self.assertEqual(self.run_main(["fetch", "-c", "--no-color"]), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

