"""Tests for the local-model clients.

A tiny threaded HTTP stub server stands in for Ollama / an OpenAI-compatible
server, so the whole HTTP layer is exercised without any external process.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcnews.llm import (
    JSONExtractionError,
    LLMError,
    LLMUnavailable,
    OpenAICompatClient,
    OllamaClient,
    balanced_slice,
    extract_json,
    http_json,
    strip_code_fences,
)


class StubHandler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, object]] = {}
    requests: list[dict] = []

    def _respond(self) -> None:
        status, payload = self.routes.get(self.path, (404, {"error": "unknown path"}))
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        StubHandler.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        StubHandler.requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(raw) if raw else None,
            }
        )
        self._respond()

    def log_message(self, *args) -> None:  # keep the test output clean
        return


class StubServerMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = "http://127.0.0.1:%d" % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        StubHandler.routes = {}
        StubHandler.requests = []


class JsonExtractionTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_inside_prose(self):
        self.assertEqual(extract_json('Sure, here you go: {"a": {"b": 2}} hope it helps'), {"a": {"b": 2}})

    def test_trailing_comma_is_repaired(self):
        self.assertEqual(extract_json('{"a": [1, 2,],}'), {"a": [1, 2]})

    def test_array_is_supported(self):
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_braces_inside_strings_do_not_confuse_it(self):
        self.assertEqual(extract_json('{"a": "} not the end {"}'), {"a": "} not the end {"})

    def test_empty_and_plain_text_raise(self):
        with self.assertRaises(JSONExtractionError):
            extract_json("")
        with self.assertRaises(JSONExtractionError):
            extract_json("I cannot analyse this.")

    def test_strip_code_fences(self):
        self.assertEqual(strip_code_fences("```json\n{}\n```"), "{}")
        self.assertEqual(strip_code_fences("plain"), "plain")

    def test_balanced_slice(self):
        self.assertEqual(balanced_slice('{"a": 1} tail', 0), '{"a": 1}')
        self.assertIsNone(balanced_slice('{"a": 1', 0))
        self.assertIsNone(balanced_slice("abc", 0))


class OllamaClientTests(StubServerMixin, unittest.TestCase):
    def client(self, model: str = "qwen2.5:3b") -> OllamaClient:
        return OllamaClient(model=model, base_url=self.base_url, timeout=5)

    def test_list_models(self):
        StubHandler.routes["/api/tags"] = (200, {"models": [{"name": "qwen2.5:3b"}, {"name": "qwen3:4b"}]})
        self.assertEqual(self.client().list_models(), ["qwen2.5:3b", "qwen3:4b"])

    def test_health_ok(self):
        StubHandler.routes["/api/tags"] = (200, {"models": [{"name": "qwen2.5:3b"}]})
        healthy, message = self.client().health()
        self.assertTrue(healthy)
        self.assertIn("qwen2.5:3b", message)

    def test_health_reports_missing_model(self):
        StubHandler.routes["/api/tags"] = (200, {"models": [{"name": "llama3:8b"}]})
        healthy, message = self.client().health()
        self.assertFalse(healthy)
        self.assertIn("not installed", message)

    def test_health_reports_unreachable_server(self):
        healthy, message = OllamaClient("m", "http://127.0.0.1:9", timeout=2).health()
        self.assertFalse(healthy)
        self.assertTrue(message)

    def test_chat_returns_content_and_sends_expected_payload(self):
        StubHandler.routes["/api/chat"] = (200, {"message": {"role": "assistant", "content": '{"ok": true}'}})
        content = self.client().chat("system prompt", "user prompt")
        self.assertEqual(content, '{"ok": true}')

        sent = StubHandler.requests[-1]["body"]
        self.assertEqual(sent["model"], "qwen2.5:3b")
        self.assertFalse(sent["stream"])
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertEqual(sent["options"]["num_ctx"], 8192)

    def test_chat_accepts_legacy_response_field(self):
        StubHandler.routes["/api/chat"] = (200, {"response": "hello"})
        self.assertEqual(self.client().chat("a", "b"), "hello")

    def test_chat_without_content_raises(self):
        StubHandler.routes["/api/chat"] = (200, {"message": {"content": ""}})
        with self.assertRaises(LLMError):
            self.client().chat("a", "b")

    def test_server_error_becomes_llm_error(self):
        StubHandler.routes["/api/chat"] = (500, {"error": "boom"})
        with self.assertRaises(LLMError):
            self.client().chat("a", "b")

    def test_unreachable_port_raises_unavailable(self):
        client = OllamaClient("m", "http://127.0.0.1:9", timeout=2)
        with self.assertRaises(LLMUnavailable):
            client.chat("a", "b", retries=0)


class OpenAICompatClientTests(StubServerMixin, unittest.TestCase):
    def client(self, api_key: str | None = None) -> OpenAICompatClient:
        return OpenAICompatClient(model="local-model", base_url=self.base_url, timeout=5, api_key=api_key)

    def test_list_models(self):
        StubHandler.routes["/models"] = (200, {"data": [{"id": "local-model"}]})
        self.assertEqual(self.client().list_models(), ["local-model"])

    def test_chat_returns_content(self):
        StubHandler.routes["/chat/completions"] = (
            200,
            {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]},
        )
        self.assertEqual(self.client().chat("s", "u"), "hi there")

    def test_api_key_is_sent_as_bearer_header(self):
        StubHandler.routes["/chat/completions"] = (200, {"choices": [{"message": {"content": "x"}}]})
        self.client(api_key="dummy-key").chat("s", "u")
        self.assertEqual(StubHandler.requests[-1]["headers"].get("Authorization"), "Bearer dummy-key")

    def test_missing_choices_raises(self):
        StubHandler.routes["/chat/completions"] = (200, {"choices": []})
        with self.assertRaises(LLMError):
            self.client().chat("s", "u")


class HttpHelperTests(StubServerMixin, unittest.TestCase):
    def test_http_json_parses_body(self):
        StubHandler.routes["/thing"] = (200, {"a": 1})
        self.assertEqual(http_json(self.base_url + "/thing"), {"a": 1})

    def test_http_error_includes_status_and_body(self):
        StubHandler.routes["/thing"] = (418, {"error": "teapot"})
        with self.assertRaises(LLMError) as ctx:
            http_json(self.base_url + "/thing")
        self.assertIn("418", str(ctx.exception))
        self.assertIn("teapot", str(ctx.exception))


class HasModelTests(unittest.TestCase):
    def test_exact_tag_and_base_name_matches(self):
        client = OllamaClient("qwen2.5:3b", "http://127.0.0.1:1")
        self.assertTrue(client.has_model(["qwen2.5:3b"]))
        self.assertTrue(client.has_model(["Qwen2.5:3B"]))
        self.assertTrue(client.has_model(["qwen2.5:3b-instruct-q4_K_M"]))
        self.assertTrue(client.has_model(["qwen2.5:7b"]))
        self.assertFalse(client.has_model(["llama3:8b"]))
        self.assertFalse(client.has_model([]))

    def test_blank_model_never_matches(self):
        self.assertFalse(OllamaClient("  ", "http://127.0.0.1:1").has_model(["anything"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

