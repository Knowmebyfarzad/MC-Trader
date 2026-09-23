#!/usr/bin/env python
"""Web UI for McNews - wraps the CLI features (fetch / analyze / channels) in a
small local website, so non-technical users can drive the tool from a browser.

    python webserver.py                 # serves on http://127.0.0.1:8000
    python webserver.py --port 9000     # custom port
    python webserver.py --host 0.0.0.0  # expose for a Cloudflare tunnel

The page is a single static file (``web/index.html``) that talks to JSON
endpoints below. Only the Python standard library is used for serving, so no
new dependencies are added beyond telethon.

Endpoints:
    GET  /                    -> web/index.html
    GET  /api/status          -> settings, session + model health
    GET  /api/channels        -> channels the Telegram account can read
    POST /api/run             -> start a job  {mode, channels, limit, ...}
    GET  /api/job/<id>        -> poll a job's status and result
    GET  /api/reports         -> recent saved analysis reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcnews.analyzer import NewsAnalyzer
from mcnews.config import PROJECT_ROOT, Settings, split_list
from mcnews.llm import LLMError, LLMUnavailable, build_llm
from mcnews.models import AnalysisReport, NewsItem
from mcnews.store import NewsStore
from mcnews.telegram_source import (
    TelegramAuthRequired,
    TelegramError,
    TelegramNewsReader,
    TelegramNotConfigured,
    parse_channel_refs,
)

WEB_ROOT = PROJECT_ROOT / "web"
INDEX_FILE = WEB_ROOT / "index.html"

# ------------------------------------------------------------------- job store
# Jobs are kept in memory only; the UI polls them. A lock guards both the dict
# and single-job-at-a-time access to Telegram (the session file dislikes
# concurrent reads from the same account).
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
MAX_JOBS_KEPT = 25


def job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe view of a job (drops non-serialisable internals)."""
    return {
        "id": job["id"],
        "status": job["status"],  # queued | running | done | error
        "mode": job.get("mode"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "progress": job.get("progress", ""),
        "error": job.get("error"),
        "result": job.get("result"),
    }


def start_job(mode: str, params: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "running",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": "queued",
        "error": None,
        "result": None,
        "params": params,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        # prune oldest finished jobs so the dict does not grow forever
        done_ids = [
            jid
            for jid, j in sorted(JOBS.items(), key=lambda kv: kv[1]["created_at"])
            if j["status"] in ("done", "error") and jid != job_id
        ]
        for jid in done_ids[: max(0, len(JOBS) - MAX_JOBS_KEPT)]:
            JOBS.pop(jid, None)
    worker = threading.Thread(target=_run_job, args=(job,), daemon=True)
    worker.start()
    return job_id


# --------------------------------------------------------------- core pipeline
def read_channels(
    settings: Settings,
    refs: list[str],
    limit: int,
    keywords: list[str],
    only_new: bool,
) -> tuple[list[NewsItem], list[dict[str, Any]]]:
    """Fetch posts from Telegram, mirroring cli.py's ``fetch`` behaviour."""
    store = NewsStore(settings.data_dir)

    async def _gather() -> list[Any]:
        reader = TelegramNewsReader(settings)
        await reader.connect()
        try:
            return await reader.fetch_many(
                refs,
                limit=limit,
                keywords=keywords,
                state=store.state if only_new else None,
                only_new=only_new,
            )
        finally:
            await reader.close()

    reads = asyncio.run(_gather())
    items: list[NewsItem] = [item for read in reads for item in read.items]
    items.sort(key=lambda item: (item.date, item.message_id), reverse=True)
    if only_new:
        store.state.save()
    if items:
        store.save_items(items)

    read_summaries = [
        {
            "ref": read.ref,
            "ok": read.ok,
            "title": read.title,
            "kind": read.kind,
            "fetched": read.fetched,
            "kept": read.kept,
            "skipped": read.skipped,
            "filtered": read.filtered,
            "error": read.error,
        }
        for read in reads
    ]
    return items, read_summaries


def analyse_items(settings: Settings, items: list[NewsItem], focus: str) -> AnalysisReport:
    """Run the local model over the fetched posts (cli.py ``analyze``)."""
    llm = build_llm(settings)
    healthy, message = llm.health()
    if not healthy:
        raise LLMUnavailable("local model check failed: %s" % message)
    extra = ("Extra analyst focus: %s" % focus) if focus else None
    analyzer = NewsAnalyzer(
        llm,
        batch_chars=settings.batch_chars,
        max_message_chars=settings.max_message_chars,
        extra_instructions=extra,
    )
    return analyzer.analyze(items)


def _run_job(job: dict[str, Any]) -> None:
    """Worker: fetch (and optionally analyse) posts, storing the result."""
    params = job["params"]
    mode = job["mode"]
    try:
        job["progress"] = "loading configuration"
        settings = Settings.from_env()
        settings.ensure_dirs()

        raw_refs = split_list(params.get("channels") or [])
        refs, ref_errors = parse_channel_refs(raw_refs)
        if not refs:
            details = "; ".join(ref_errors) if ref_errors else "no channel given"
            raise ValueError("Please provide at least one valid channel (%s)." % details)

        limit = max(1, min(int(params.get("limit") or 20), 200))
        keywords = split_list(params.get("keywords") or [])
        focus = (params.get("focus") or "").strip()
        only_new = not bool(params.get("ignore_state"))

        job["progress"] = "connecting to Telegram and reading posts"
        items, reads = read_channels(settings, refs, limit, keywords, only_new)
        job["progress"] = "read %d post(s) from %d channel(s)" % (len(items), len(refs))

        result: dict[str, Any] = {
            "mode": mode,
            "channels": refs,
            "reads": reads,
            "items": [item.to_dict() for item in items],
        }

        if mode == "analyze":
            if not items:
                result["report"] = None
                result["note"] = (
                    "No new posts to analyse. Untick 'only new posts' to re-read "
                    "the channel history."
                )
            else:
                job["progress"] = "analysing %d post(s) with the local model" % len(items)
                report = analyse_items(settings, items, focus)
                store = NewsStore(settings.data_dir)
                report_path = store.save_report(report, tag="web")
                result["report"] = report.to_dict()
                result["report_path"] = str(report_path)

        job["result"] = result
        job["status"] = "done"
    except (TelegramNotConfigured, TelegramAuthRequired, ValueError) as exc:
        job["status"] = "error"
        job["error"] = str(exc)
    except LLMUnavailable as exc:
        job["status"] = "error"
        job["error"] = (
            "The local AI model is not running (%s). Start it, e.g. 'ollama serve'." % exc
        )
    except (TelegramError, LLMError) as exc:
        job["status"] = "error"
        job["error"] = str(exc)
    except Exception as exc:  # never leave the UI hanging
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = "unexpected error: %s: %s" % (type(exc).__name__, exc)
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["params"] = {}  # drop request payload from memory
        RUN_LOCK.release()


def list_readable_channels() -> list[dict[str, Any]]:
    """Channels/groups the logged-in account can read (for the datalist)."""
    settings = Settings.from_env()

    async def _run() -> list[Any]:
        reader = TelegramNewsReader(settings)
        await reader.connect()
        try:
            return await reader.list_dialogs(limit=300)
        finally:
            await reader.close()

    dialogs = asyncio.run(_run())
    return [dialog.__dict__ for dialog in dialogs]


def status_payload() -> dict[str, Any]:
    settings = Settings.from_env()
    healthy, message = False, "not checked"
    try:
        llm = build_llm(settings)
        healthy, message = llm.health()
    except Exception as exc:
        message = str(exc)
    return {
        "telegram_ok": not settings.telegram_problems(),
        "telegram_problems": settings.telegram_problems(),
        "session_exists": settings.session_exists(),
        "default_channels": list(settings.channels),
        "model": settings.model,
        "provider": settings.provider,
        "model_ok": healthy,
        "model_message": message,
    }


def recent_reports(limit: int = 10) -> list[dict[str, Any]]:
    """Summaries of the most recent saved analysis reports."""
    settings = Settings.from_env()
    analyses_dir = settings.data_dir / "analyses"
    reports: list[dict[str, Any]] = []
    if not analyses_dir.is_dir():
        return reports
    files = sorted(analyses_dir.glob("analysis-*.json"), reverse=True)[:limit]
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overall = data.get("overall") or {}
        reports.append(
            {
                "file": path.name,
                "generated_at": data.get("generated_at"),
                "model": data.get("model"),
                "item_count": data.get("item_count"),
                "sentiment": overall.get("sentiment"),
                "market_impact": overall.get("market_impact"),
                "summary": str(overall.get("summary") or "")[:280],
            }
        )
    return reports


# ------------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    server_version = "McNewsWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # quieter console
        print("[web] %s - %s" % (self.address_string(), format % args))

    # ------------------------------------------------------------- utilities
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_file(INDEX_FILE, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(status_payload())
            elif path == "/api/reports":
                self._send_json({"reports": recent_reports()})
            elif path == "/api/channels":
                self._send_json({"channels": list_readable_channels()})
            elif path.startswith("/api/job/"):
                job_id = path.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job is None:
                    self._send_json({"error": "job not found"}, status=404)
                else:
                    self._send_json(job_snapshot(job))
            else:
                self._send_json({"error": "not found"}, status=404)
        except (TelegramNotConfigured, TelegramAuthRequired) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": "%s: %s" % (type(exc).__name__, exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/run":
                params = self._read_json_body()
                mode = (params.get("mode") or "").strip().lower()
                if mode not in ("fetch", "analyze"):
                    self._send_json({"error": "mode must be 'fetch' or 'analyze'"}, status=400)
                    return
                if not split_list(params.get("channels") or []):
                    self._send_json({"error": "enter a channel name first"}, status=400)
                    return
                # one job at a time: the Telegram session file is not concurrent-safe
                if not RUN_LOCK.acquire(blocking=False):
                    self._send_json(
                        {"error": "another run is in progress - wait for it to finish"},
                        status=429,
                    )
                    return
                job_id = start_job(mode, params)  # RUN_LOCK released in _run_job
                self._send_json({"job_id": job_id})
            else:
                self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": "%s: %s" % (type(exc).__name__, exc)}, status=500)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="McNews web UI server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (use 0.0.0.0 when sharing via a Cloudflare tunnel)",
    )
    parser.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    args = parser.parse_args(argv)

    if not INDEX_FILE.is_file():
        print("missing %s" % INDEX_FILE)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d" % (args.host, args.port)
    print("McNews web UI running on %s  (Ctrl+C to stop)" % url)
    print("For a Cloudflare quick tunnel:  cloudflared tunnel --url %s" % url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

