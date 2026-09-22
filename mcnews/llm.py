"""Clients for a *local* chat model.

Two providers are supported out of the box:

* ``ollama`` -> ``POST {base}/api/chat``            (Ollama, default)
* ``openai`` -> ``POST {base}/chat/completions``    (LM Studio, llama.cpp, vLLM, ...)

Only the standard library is used for HTTP, so the MVP has a single
third-party dependency (telethon).
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class LLMError(RuntimeError):
    """The model server answered, but with an error or useless content."""


class LLMUnavailable(LLMError):
    """The model server could not be reached (not started / wrong port)."""


class JSONExtractionError(LLMError):
    """The model answered with something that is not parsable JSON."""


# --------------------------------------------------------------------------- http
def http_request(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> str:
    """Minimal JSON-over-HTTP helper returning the decoded response body."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:  # server answered with 4xx/5xx
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # pragma: no cover - body may be unreadable
            detail = "<no body>"
        finally:
            exc.close()
        raise LLMError("HTTP %s from %s: %s" % (exc.code, url, detail.strip())) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise LLMUnavailable("cannot reach %s (%s)" % (url, reason)) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise LLMUnavailable("timed out after %ss waiting for %s" % (timeout, url)) from exc


def http_json(url: str, **kwargs: Any) -> Any:
    body = http_request(url, **kwargs)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError("invalid JSON from %s: %s" % (url, body[:200])) from exc


# ------------------------------------------------------------------- json parsing
def strip_code_fences(text: str) -> str:
    match = CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def balanced_slice(text: str, start: int) -> str | None:
    """Return the balanced ``{...}``/``[...]`` block starting at ``start``."""
    if start < 0 or start >= len(text) or text[start] not in "{[":
        return None
    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> Any:
    """Best-effort extraction of one JSON object/array from a model answer.

    Handles clean JSON, fenced JSON, JSON surrounded by prose and trailing
    commas. Raises :class:`JSONExtractionError` when nothing usable is found.
    """
    if not text or not text.strip():
        raise JSONExtractionError("the model returned an empty response")

    candidates: list[str] = []
    fenced = strip_code_fences(text)
    for candidate in (fenced, text.strip()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        for blob in (candidate, TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                continue

    for opener in ("{", "["):
        index = fenced.find(opener)
        while index != -1:
            block = balanced_slice(fenced, index)
            if block:
                for blob in (block, TRAILING_COMMA_RE.sub(r"\1", block)):
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        pass
            index = fenced.find(opener, index + 1)

    raise JSONExtractionError("no JSON object found in model response: %s" % text[:200])


# ----------------------------------------------------------------------- clients
class LocalLLMClient:
    """Common behaviour for local chat-model providers."""

    provider = "local"

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 240.0,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.api_key = api_key

    # ------------------------------------------------------------------ info
    def describe(self) -> str:
        return "%s (%s) at %s" % (self.model, self.provider, self.base_url)

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        return headers

    # ------------------------------------------------------------------ chat
    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def chat(self, system_prompt: str, user_prompt: str, retries: int = 1) -> str:
        """Call the model. Only transport failures are retried."""
        last_error: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            try:
                return self._chat_once(system_prompt, user_prompt)
            except LLMUnavailable as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise LLMError("model call failed")

    # ------------------------------------------------------------------ models
    def list_models(self) -> list[str]:
        return []

    def has_model(self, models: list[str]) -> bool:
        wanted = (self.model or "").strip().lower()
        if not wanted:
            return False
        for name in models:
            candidate = name.strip().lower()
            if candidate == wanted or candidate.startswith(wanted + ":") or candidate.startswith(wanted + "@"):
                return True
            if candidate.split(":")[0] == wanted.split(":")[0]:
                return True
        return False

    def health(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except LLMError as exc:
            return False, str(exc)
        if not models:
            return True, "server reachable at %s (no model list reported)" % self.base_url
        if self.has_model(models):
            return True, "server reachable at %s, model '%s' found" % (self.base_url, self.model)
        return False, "server reachable but model '%s' is not installed (available: %s)" % (
            self.model,
            ", ".join(models[:12]),
        )


class OllamaClient(LocalLLMClient):
    """Ollama native API (``POST /api/chat``)."""

    provider = "ollama"

    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        data = http_json(self.base_url + "/api/chat", method="POST", payload=payload, timeout=self.timeout)
        content = ""
        if isinstance(data, dict):
            message = data.get("message") or {}
            content = (message.get("content") or data.get("response") or "").strip()
        if not content:
            raise LLMError("ollama returned no content for model '%s'" % self.model)
        return content

    def list_models(self) -> list[str]:
        data = http_json(self.base_url + "/api/tags", timeout=15)
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        names: list[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if name:
                    names.append(str(name))
        return names


class OpenAICompatClient(LocalLLMClient):
    """Any OpenAI-compatible local server (``POST /chat/completions``)."""

    provider = "openai"

    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        data = http_json(
            self.base_url + "/chat/completions",
            method="POST",
            payload=payload,
            headers=self.headers(),
            timeout=self.timeout,
        )
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise LLMError("no choices returned by %s" % self.base_url)
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise LLMError("empty content returned by %s" % self.base_url)
        return content

    def list_models(self) -> list[str]:
        data = http_json(self.base_url + "/models", headers=self.headers(), timeout=15)
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        return [str(e.get("id")) for e in entries if isinstance(e, dict) and e.get("id")]


def build_llm(settings: Any) -> LocalLLMClient:
    """Create the configured client from a :class:`mcnews.config.Settings`."""
    kwargs: dict[str, Any] = dict(
        model=settings.model,
        base_url=settings.resolved_base_url(),
        timeout=float(settings.timeout),
        temperature=float(settings.temperature),
        num_ctx=int(settings.num_ctx),
        max_tokens=int(settings.max_tokens),
        api_key=settings.api_key,
    )
    if getattr(settings, "provider", "ollama") == "openai":
        return OpenAICompatClient(**kwargs)
    return OllamaClient(**kwargs)
