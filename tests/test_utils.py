"""Tests for the small shared helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mcnews.utils import (
    as_utc,
    chunk_seq,
    clean_text,
    coerce_float,
    coerce_int,
    coerce_sentence_list,
    coerce_str_list,
    dedupe,
    fmt_dt,
    parse_bool,
    shorten,
    truncate,
    unique_sorted,
)


class CleanTextTests(unittest.TestCase):
    def test_collapses_whitespace_and_control_characters(self):
        raw = "  hello\x00  world \u200b\n\n\n\ntail  "
        self.assertEqual(clean_text(raw), "hello world\n\ntail")

    def test_handles_none_and_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")

    def test_trims_each_line_and_normalises_newlines(self):
        self.assertEqual(clean_text("a  \r\n   b"), "a\nb")


class TruncateTests(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(truncate("abc", 10), "abc")

    def test_long_text_marked(self):
        self.assertEqual(truncate("abcdef", 3), "abc ...[truncated]")

    def test_non_positive_budget(self):
        self.assertEqual(truncate("abcdef", 0), "...[truncated]")

    def test_shorten_uses_ellipsis(self):
        self.assertEqual(shorten("hello world", 8), "hello...")
        self.assertEqual(shorten("hello world", 11), "hello world")


class BoolTests(unittest.TestCase):
    def test_truthy_and_falsy_words(self):
        for token in ("1", "true", "YES", "on", "enabled", True):
            self.assertTrue(parse_bool(token), token)
        for token in ("0", "false", "No", "off", "disabled", False):
            self.assertFalse(parse_bool(token), token)

    def test_unknown_uses_default(self):
        self.assertTrue(parse_bool("maybe", default=True))
        self.assertFalse(parse_bool("maybe"))
        self.assertFalse(parse_bool(None))


class ChunkTests(unittest.TestCase):
    def test_groups_respect_budget(self):
        items = [1, 2, 3, 4, 5]
        chunks = chunk_seq(items, size_of=lambda _: 5, budget=12)
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_oversized_item_gets_its_own_chunk(self):
        chunks = chunk_seq(["a", "b"], size_of=lambda value: 100 if value == "a" else 1, budget=10)
        self.assertEqual(chunks, [["a"], ["b"]])

    def test_empty_input(self):
        self.assertEqual(chunk_seq([], size_of=len, budget=10), [])


class CoercionTests(unittest.TestCase):
    def test_coerce_int(self):
        self.assertEqual(coerce_int("7"), 7)
        self.assertEqual(coerce_int(7.8), 7)
        self.assertEqual(coerce_int("abc", default=3), 3)
        self.assertIsNone(coerce_int(None))
        self.assertIsNone(coerce_int(True))

    def test_coerce_int_clamps(self):
        self.assertEqual(coerce_int(-5, None, 1, 10), 1)
        self.assertEqual(coerce_int(50, None, 1, 10), 10)

    def test_coerce_float(self):
        self.assertEqual(coerce_float("0.5"), 0.5)
        self.assertEqual(coerce_float("nope", default=1.5), 1.5)

    def test_coerce_str_list(self):
        self.assertEqual(coerce_str_list("gold, usd; btc"), ["btc", "gold", "usd"])
        self.assertEqual(coerce_str_list(["b", "a", "b"]), ["a", "b"])
        self.assertEqual(coerce_str_list(None), [])

    def test_coerce_sentence_list_keeps_commas(self):
        self.assertEqual(coerce_sentence_list("risky, but possible"), ["risky, but possible"])
        self.assertEqual(coerce_sentence_list(["a", " b ", "a"]), ["a", "b"])
        self.assertEqual(coerce_sentence_list("one\n two"), ["one", "two"])
        self.assertEqual(coerce_sentence_list(" - dashes stripped"), ["dashes stripped"])
        self.assertEqual(coerce_sentence_list(None), [])


class MiscTests(unittest.TestCase):
    def test_as_utc_assumes_utc_for_naive(self):
        naive = datetime(2026, 1, 2, 3, 4)
        self.assertEqual(as_utc(naive).tzinfo, timezone.utc)
        self.assertEqual(as_utc(None).tzinfo, timezone.utc)
        aware = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        self.assertEqual(as_utc(aware), aware)

    def test_fmt_dt(self):
        self.assertEqual(fmt_dt(datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)), "2026-01-02 03:04")
        self.assertEqual(fmt_dt(None), "-")

    def test_dedupe_preserves_order(self):
        self.assertEqual(dedupe(["b", "a", "b", "c"], key=lambda value: value), ["b", "a", "c"])

    def test_unique_sorted(self):
        self.assertEqual(unique_sorted([" b", "a", "a", ""]), ["a", "b"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
