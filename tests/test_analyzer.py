"""Tests for batching, output normalisation and merging (the LLM is faked)."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from mcnews.analyzer import (
    NewsAnalyzer,
    merge_overalls,
    normalize_horizon,
    normalize_item,
    normalize_overall,
    normalize_sentiment,
    report_summary_line,
    synthesize_overall_from_items,
)
from mcnews.llm import LLMUnavailable, LocalLLMClient
from mcnews.models import NewsItem


class FakeLLM(LocalLLMClient):
    """LocalLLMClient with the HTTP call replaced by a queue of answers."""

    provider = "fake"

    def __init__(self, responses):
        super().__init__(model="fake-model", base_url="http://fake.local")
        self.responses = list(responses)
        self.calls: list[str] = []

    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(user_prompt)
        if not self.responses:
            return "{}"
        answer = self.responses[0]
        if isinstance(answer, Exception):
            raise answer  # stays queued: an outage also fails the retry
        self.responses.pop(0)
        return answer


def make_item(
    message_id: int,
    text: str = "market news text that is long enough",
    channel: str = "c",
    minutes: int = 0,
) -> NewsItem:
    return NewsItem(
        channel=channel,
        message_id=message_id,
        text=text,
        date=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes),
        channel_title="Desk",
    )


def good_answer(**overrides) -> str:
    payload = {
        "overall": {
            "summary": "Calm session.",
            "sentiment": "bullish",
            "confidence": 0.7,
            "market_impact": 4,
            "tradable": True,
            "time_horizon": "days",
            "key_drivers": ["rates unchanged"],
        },
        "items": [
            {
                "channel": "c",
                "message_id": 1,
                "summary": "Rates unchanged.",
                "sentiment": "bullish",
                "impact": 4,
                "assets": ["gold"],
                "topics": ["macro"],
                "actionable": True,
            }
        ],
        "watchlist": ["next policy meeting"],
        "risks": ["thin liquidity, but manageable"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class NormalisationTests(unittest.TestCase):
    def test_sentiment_aliases(self):
        self.assertEqual(normalize_sentiment("Bull"), "bullish")
        self.assertEqual(normalize_sentiment("NEGATIVE"), "bearish")
        self.assertEqual(normalize_sentiment("sideways-ish"), "mixed")
        self.assertEqual(normalize_sentiment(None), "mixed")

    def test_horizon_normalisation(self):
        self.assertEqual(normalize_horizon("Weekly"), "weeks")
        self.assertEqual(normalize_horizon("intra-day"), "intraday")
        self.assertEqual(normalize_horizon("years"), "unspecified")

    def test_overall_is_clamped(self):
        overall = normalize_overall(
            {"summary": " s ", "sentiment": "Bull", "confidence": 1.5, "market_impact": 0, "tradable": "yes"}
        )
        self.assertEqual(overall["summary"], "s")
        self.assertEqual(overall["sentiment"], "bullish")
        self.assertEqual(overall["confidence"], 1.0)
        self.assertEqual(overall["market_impact"], 1)
        self.assertTrue(overall["tradable"])
        self.assertEqual(overall["time_horizon"], "unspecified")
        self.assertEqual(overall["key_drivers"], [])

    def test_item_normalisation(self):
        entry = normalize_item(
            {
                "channel": "c",
                "message_id": "7",
                "summary": "text",
                "sentiment": "up",
                "impact": "15",
                "assets": "gold, usd",
                "topics": ["macro", "macro"],
                "actionable": "no",
            }
        )
        self.assertEqual(entry["message_id"], 7)
        self.assertEqual(entry["sentiment"], "bullish")
        self.assertEqual(entry["impact"], 10)
        self.assertEqual(entry["assets"], ["gold", "usd"])
        self.assertEqual(entry["topics"], ["macro"])
        self.assertFalse(entry["actionable"])

    def test_unusable_entries_are_dropped(self):
        self.assertIsNone(normalize_item(None))
        self.assertIsNone(normalize_item("just a string"))
        self.assertIsNone(normalize_item({"summary": ""}))
        self.assertIsNotNone(normalize_item({"message_id": 3}))

    def test_merge_single_overall_is_a_copy(self):
        only = normalize_overall({"summary": "a", "sentiment": "bearish", "market_impact": 9})
        merged = merge_overalls([only])
        self.assertEqual(merged, only)
        self.assertIsNot(merged, only)

    def test_merge_majority_max_and_any(self):
        merged = merge_overalls(
            [
                {
                    "summary": "a",
                    "sentiment": "bullish",
                    "confidence": 0.4,
                    "market_impact": 3,
                    "tradable": False,
                    "time_horizon": "days",
                    "key_drivers": ["x"],
                },
                {
                    "summary": "b",
                    "sentiment": "bullish",
                    "confidence": 0.8,
                    "market_impact": 7,
                    "tradable": True,
                    "time_horizon": "days",
                    "key_drivers": ["y"],
                },
                {
                    "summary": "c",
                    "sentiment": "bearish",
                    "confidence": 0.6,
                    "market_impact": 2,
                    "tradable": False,
                    "time_horizon": "weeks",
                    "key_drivers": ["x"],
                },
            ]
        )
        self.assertEqual(merged["sentiment"], "bullish")
        self.assertEqual(merged["market_impact"], 7)
        self.assertTrue(merged["tradable"])
        self.assertEqual(merged["confidence"], 0.6)
        self.assertEqual(merged["time_horizon"], "days")
        self.assertEqual(merged["key_drivers"], ["x", "y"])
        self.assertEqual(merged["batches_merged"], 3)
        self.assertIn("a", merged["summary"])

    def test_merge_tie_becomes_mixed(self):
        merged = merge_overalls([{"sentiment": "bullish"}, {"sentiment": "bearish"}])
        self.assertEqual(merged["sentiment"], "mixed")

    def test_merge_without_data(self):
        self.assertEqual(merge_overalls([]), {})

    def test_derived_overall_when_model_omits_it(self):
        answer = json.dumps(
            {
                "items": [
                    {"channel": "c", "message_id": 1, "summary": "a", "sentiment": "bearish", "impact": 6},
                    {"channel": "c", "message_id": 2, "summary": "b", "sentiment": "bearish", "impact": 3},
                ]
            }
        )
        report = NewsAnalyzer(FakeLLM([answer])).analyze([make_item(1), make_item(2)])
        self.assertTrue(report.overall["derived"])
        self.assertEqual(report.overall["sentiment"], "bearish")
        self.assertEqual(report.overall["market_impact"], 6)
        self.assertEqual(len(report.items), 2)

    def test_derived_overall_is_mixed_on_a_tie(self):
        items = [
            {"channel": "c", "message_id": 1, "summary": "a", "sentiment": "bullish", "impact": 4},
            {"channel": "c", "message_id": 2, "summary": "b", "sentiment": "bearish", "impact": 2},
        ]
        derived = synthesize_overall_from_items(items)
        self.assertEqual(derived["sentiment"], "mixed")
        self.assertEqual(derived["market_impact"], 4)
        self.assertFalse(derived["tradable"])

    def test_derived_overall_for_no_items(self):
        self.assertEqual(synthesize_overall_from_items([]), {})


class PrepareTests(unittest.TestCase):
    def test_filters_short_and_duplicate_posts(self):
        analyzer = NewsAnalyzer(FakeLLM([]))
        prepared = analyzer.prepare(
            [
                make_item(1),
                make_item(1),  # duplicate uid
                make_item(2, text=""),  # not usable
                NewsItem(channel="c", message_id=3, text="ok text"),
            ]
        )
        self.assertEqual([item.message_id for item in prepared], [3, 1])

    def test_newest_first(self):
        analyzer = NewsAnalyzer(FakeLLM([]))
        prepared = analyzer.prepare([make_item(1, minutes=30), make_item(2, minutes=1)])
        self.assertEqual([item.message_id for item in prepared], [2, 1])

    def test_skip_uids(self):
        analyzer = NewsAnalyzer(FakeLLM([]))
        prepared = analyzer.prepare([make_item(1), make_item(2)], skip_uids={"c:1"})
        self.assertEqual([item.message_id for item in prepared], [2])


class AnalyzeTests(unittest.TestCase):
    def test_single_batch_report(self):
        llm = FakeLLM([good_answer()])
        report = NewsAnalyzer(llm).analyze([make_item(1)])
        self.assertEqual(report.provider, "fake")
        self.assertEqual(report.model, "fake-model")
        self.assertEqual(report.item_count, 1)
        self.assertEqual(report.batch_count, 1)
        self.assertEqual(report.parsed_batches, 1)
        self.assertEqual(report.overall["sentiment"], "bullish")
        self.assertEqual(report.items[0]["message_id"], 1)
        self.assertEqual(report.items[0]["batch"], 1)
        self.assertEqual(report.watchlist, ["next policy meeting"])
        self.assertEqual(report.risks, ["thin liquidity, but manageable"])
        self.assertEqual(report.sources, ["Desk"])
        self.assertEqual(len(llm.calls), 1)

    def test_prompt_contains_contract_and_posts(self):
        llm = FakeLLM([good_answer()])
        NewsAnalyzer(llm, extra_instructions="Extra analyst focus: gold").analyze([make_item(1, text="gold rallied")])
        prompt = llm.calls[0]
        self.assertIn("POSTS:", prompt)
        self.assertIn("message_id", prompt)
        self.assertIn("Extra analyst focus: gold", prompt)
        self.assertIn("gold rallied", prompt)
        self.assertIn("channel=c", prompt)

    def test_no_usable_posts_skips_the_model(self):
        llm = FakeLLM([])
        report = NewsAnalyzer(llm).analyze([make_item(1, text="")])
        self.assertEqual(llm.calls, [])
        self.assertEqual(report.item_count, 0)
        self.assertIn("No usable posts", report.overall["summary"])

    def test_unparsable_answer_is_recorded_not_fatal(self):
        llm = FakeLLM(["I am sorry, I cannot do that."])
        with self.assertLogs("mcnews.analyzer", level="WARNING"):
            report = NewsAnalyzer(llm).analyze([make_item(1)])
        self.assertEqual(report.parsed_batches, 0)
        self.assertEqual(report.failed_batches, 1)
        self.assertTrue(report.raw_responses[0].startswith("[unparsed batch 1]"))

    def test_non_object_json_is_recorded(self):
        llm = FakeLLM(["[1, 2, 3]"])
        with self.assertLogs("mcnews.analyzer", level="WARNING"):
            report = NewsAnalyzer(llm).analyze([make_item(1)])
        self.assertEqual(report.failed_batches, 1)
        self.assertTrue(report.raw_responses[0].startswith("[unexpected JSON batch 1]"))

    def test_multiple_batches_are_merged(self):
        llm = FakeLLM(
            [
                good_answer(),
                good_answer(
                    overall={
                        "summary": "Second batch.",
                        "sentiment": "bearish",
                        "market_impact": 9,
                        "tradable": False,
                    }
                ),
            ]
        )
        analyzer = NewsAnalyzer(llm, batch_chars=500)
        report = analyzer.analyze([make_item(index) for index in range(1, 5)])
        self.assertEqual(report.batch_count, 2)
        self.assertEqual(report.parsed_batches, 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(report.overall["market_impact"], 9)
        self.assertTrue(report.overall["tradable"])
        self.assertEqual(report.overall["batches_merged"], 2)
        self.assertIn("Batch 2 of 2", llm.calls[1])

    def test_transport_failure_propagates(self):
        llm = FakeLLM([LLMUnavailable("server down")])
        with self.assertRaises(LLMUnavailable):
            NewsAnalyzer(llm).analyze([make_item(1)])

    def test_summary_line_mentions_model_and_sentiment(self):
        report = NewsAnalyzer(FakeLLM([good_answer()])).analyze([make_item(1)])
        line = report_summary_line(report)
        self.assertIn("fake-model", line)
        self.assertIn("sentiment=bullish", line)
        self.assertIn("posts=1", line)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


