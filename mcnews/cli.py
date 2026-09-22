"""Command line interface.

    python main.py doctor      # is everything wired up?
    python main.py login       # one-off interactive Telegram login
    python main.py channels    # what can this account read?
    python main.py fetch       # read posts only -> proves channel access
    python main.py analyze     # read posts + local model analysis
    python main.py watch       # poll + analyse new posts
    python main.py demo        # synthetic posts + local model (no Telegram)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__, console
from .analyzer import NewsAnalyzer, report_summary_line
from .config import PROJECT_ROOT, SUPPORTED_PROVIDERS, Settings, split_list
from .demo_data import build_sample_items
from .llm import LLMError, LLMUnavailable, build_llm
from .models import AnalysisReport, NewsItem
from .store import NewsStore
from .telegram_source import (
    TELEGRAM_IMPORT_ERROR,
    TelegramAuthRequired,
    TelegramError,
    TelegramNewsReader,
    TelegramNotConfigured,
    parse_channel_refs,
)
from .utils import shorten

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_USAGE = 2

EPILOG = """examples (quote the channel ref in PowerShell!):
  python main.py fetch -c "@channelname" -n 10
  python main.py fetch -c "https://t.me/somechannel" -k gold -k dollar
  python main.py analyze -c "@channelname" -n 15 --focus "gold, FX, crypto"
  python main.py analyze --from-raw
  python main.py watch -c "@channelname" --interval 60
  python main.py demo
"""


class UsageError(RuntimeError):
    """Bad combination of flags / missing configuration (exit code 2)."""


def add_common_args(parser: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    """Global flags: defined on the main parser *and* on every subparser.

    ``suppress_defaults=True`` (used for the subparser copies) is essential and
    subtle: the main parser and a subparser must not share one action object,
    because ``set_defaults``/``parents=`` reuse would let the subparser overwrite
    a value given *before* the command - e.g. ``mcnews --env custom.env fetch``.
    """
    text_default = argparse.SUPPRESS if suppress_defaults else None
    flag_default = argparse.SUPPRESS if suppress_defaults else False
    parser.add_argument(
        "--env",
        metavar="PATH",
        default=text_default,
        help="path to a .env file (default: <project>/.env)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=flag_default,
        help="debug logging (includes telethon logs)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=flag_default,
        help="disable ANSI colours",
    )


def add_channel_args(parser: argparse.ArgumentParser) -> None:
    # nargs="?" is deliberate: PowerShell drops a bare "@word" argument, and a
    # friendly "no channels given" message beats an argparse usage dump.
    parser.add_argument(
        "-c",
        "--channels",
        action="append",
        nargs="?",
        const="",
        default=[],
        metavar="REF",
        help='@name, https://t.me/name or numeric id (repeatable; quote it in PowerShell: -c "@name")',
    )
    parser.add_argument("-n", "--limit", type=int, default=20, help="max posts read per channel (default 20)")
    parser.add_argument(
        "-k",
        "--keyword",
        action="append",
        default=[],
        metavar="WORD",
        help="keep only posts containing WORD (repeatable, OR)",
    )
    parser.add_argument("--no-state", action="store_true", help="ignore data/state.json, read the newest posts")
    parser.add_argument("--json", action="store_true", help="print machine readable JSON")
    parser.add_argument("--no-save", action="store_true", help="do not write anything under data/")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", metavar="NAME", help="model name (overrides LLM_MODEL)")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="local server flavour (LLM_PROVIDER)")
    parser.add_argument("--base-url", metavar="URL", help="server url (overrides LLM_BASE_URL)")
    parser.add_argument("--batch-chars", type=int, help="max characters sent to the model per call")
    parser.add_argument(
        "--focus",
        metavar="TEXT",
        help='extra instructions for the model, e.g. "focus on gold and USD/IRR"',
    )
    parser.add_argument("--show-raw", action="store_true", help="also dump the raw model answers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcnews",
        description="Read news from Telegram channels and analyse it with a local AI model.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="mcnews %s" % __version__)
    add_common_args(parser)  # real defaults: usable before the command
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def sub(name: str, help_text: str) -> argparse.ArgumentParser:
        """A subcommand that also accepts the global flags placed after it."""
        created = subparsers.add_parser(name, help=help_text)
        add_common_args(created, suppress_defaults=True)
        return created

    sub("doctor", "check python, credentials, session, channels and local model")

    sub("login", "create the Telegram session (interactive: phone, code, 2FA password)")

    p_channels = sub("channels", "list the channels/groups this account can read")
    p_channels.add_argument("--limit", type=int, default=300, help="max dialogs to scan (default 300)")
    p_channels.add_argument("--json", action="store_true", help="print machine readable JSON")

    p_fetch = sub("fetch", "read posts only, no AI (fastest way to test channel access)")
    add_channel_args(p_fetch)
    p_fetch.add_argument("--show", type=int, default=10, help="print the first N posts (0 = all, default 10)")

    p_analyze = sub("analyze", "read posts and analyse them with the local model")
    add_channel_args(p_analyze)
    add_llm_args(p_analyze)
    p_analyze.add_argument(
        "--from-raw",
        nargs="?",
        const="latest",
        metavar="FILE",
        help="analyse a stored raw JSONL file instead of Telegram (default: newest raw file)",
    )

    p_watch = sub("watch", "poll the channels and analyse new posts as they arrive")
    add_channel_args(p_watch)
    add_llm_args(p_watch)
    p_watch.add_argument("--interval", type=int, default=60, help="seconds between polls (default 60)")
    p_watch.add_argument("--iterations", type=int, default=0, help="number of polls, 0 = until Ctrl+C (default 0)")

    p_demo = sub("demo", "run the analysis on synthetic sample posts (no Telegram needed)")
    add_llm_args(p_demo)
    p_demo.add_argument("--no-save", action="store_true", help="do not write anything under data/")
    p_demo.add_argument("--json", action="store_true", help="print machine readable JSON")

    return parser


# --------------------------------------------------------------------- helpers
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for name in ("telethon", "telethon.client", "telethon.network", "telethon.extensions"):
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.ERROR)


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Settings from .env, with CLI flags taking precedence."""
    env_file: Path | None = None
    if getattr(args, "env", None):
        env_file = Path(args.env).expanduser()
        if not env_file.is_file():
            raise UsageError("env file not found: %s" % env_file)
    settings = Settings.from_env(env_file)
    if getattr(args, "model", None):
        settings.model = args.model
    if getattr(args, "provider", None):
        settings.provider = args.provider
    if getattr(args, "base_url", None):
        settings.base_url = args.base_url
    if getattr(args, "batch_chars", None):
        settings.batch_chars = int(args.batch_chars)
    settings.ensure_dirs()
    return settings


def resolve_channel_refs(args: argparse.Namespace, settings: Settings) -> list[str]:
    raw: list[str] = []
    for value in list(getattr(args, "channels", None) or []) + list(settings.channels):
        raw.extend(split_list(value))
    refs, errors = parse_channel_refs(raw)
    for bad in errors:
        console.warn("ignoring channel reference: %s" % bad)
    if not refs:
        raise UsageError(
            "no channels given. Use -c \"@channelname\" (repeatable) or TELEGRAM_CHANNELS=@a,@b in .env.\n"
            "          PowerShell note: quote the reference - an unquoted @word is swallowed by the shell.\n"
            "          Tip: 'python main.py channels' lists every channel this account can read."
        )
    return refs


def make_analyzer(args: argparse.Namespace, settings: Settings) -> NewsAnalyzer:
    llm = build_llm(settings)
    healthy, message = llm.health()
    if healthy:
        console.ok("local model: %s" % message)
    else:
        console.warn("local model check: %s" % message)
    extra = None
    if getattr(args, "focus", None):
        extra = "Extra analyst focus: %s" % args.focus
    return NewsAnalyzer(
        llm,
        batch_chars=settings.batch_chars,
        max_message_chars=settings.max_message_chars,
        extra_instructions=extra,
    )


def dump_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# -------------------------------------------------------------------- printers
def print_reads(reads: Sequence[Any]) -> None:
    console.head("Channel read summary")
    for read in reads:
        line = read.describe()
        if read.ok:
            console.ok(line)
        else:
            console.err(line)


def print_items(items: Sequence[NewsItem], show: int = 10, header: str | None = None) -> None:
    if header:
        console.head(header)
    limit = len(items) if show <= 0 else min(show, len(items))
    for item in items[:limit]:
        console.rule()
        print(console.style(item.label_line(), "bold"))
        print(item.text if len(item.text) <= 800 else item.text[:800] + " ...")
    if limit < len(items):
        console.info("... %d more post(s) not printed (--show 0 prints everything)" % (len(items) - limit))


def print_report(
    report: AnalysisReport,
    context: dict[str, NewsItem] | None = None,
    show_raw: bool = False,
    max_items: int = 15,
) -> None:
    overall = report.overall or {}
    console.rule("=")
    console.head("ANALYSIS  %s" % report_summary_line(report))
    if report.sources:
        console.info("sources: %s" % "; ".join(report.sources))

    summary = str(overall.get("summary") or "").strip()
    if summary:
        print()
        print(summary)
    if overall.get("key_drivers"):
        console.field("key drivers", ", ".join(overall["key_drivers"]))
    if overall.get("confidence") is not None:
        console.field("confidence", overall["confidence"])
    if overall.get("batches_merged"):
        console.field("batches merged", overall["batches_merged"])

    scored = report.items or []
    if scored:
        console.rule()
        console.head("Per post (highest impact first)")
        ordered = sorted(scored, key=lambda e: (-int(e.get("impact") or 0), str(e.get("channel") or "")))
        for entry in ordered[:max_items]:
            flags: list[str] = []
            if entry.get("actionable"):
                flags.append("ACTIONABLE")
            if entry.get("assets"):
                flags.append("assets=" + ",".join(entry["assets"][:5]))
            if entry.get("topics"):
                flags.append("topics=" + ",".join(entry["topics"][:5]))
            print(
                "  %s impact=%-2s channel=%s id=%s%s"
                % (
                    console.style(shorten(str(entry.get("sentiment") or "?"), 8).ljust(8), "bold"),
                    entry.get("impact"),
                    entry.get("channel"),
                    entry.get("message_id"),
                    (" | " + " ".join(flags)) if flags else "",
                )
            )
            if entry.get("summary"):
                print("      %s" % entry["summary"])
            item = (context or {}).get("%s:%s" % (entry.get("channel"), entry.get("message_id")))
            if item is not None:
                bits = [item.date.strftime("%Y-%m-%d %H:%M UTC")]
                if item.label:
                    bits.append(item.label)
                if item.permalink:
                    bits.append(item.permalink)
                print("      " + console.style(" | ".join(bits), "dim"))
        if len(ordered) > max_items:
            console.info("... %d more scored post(s) inside the saved report" % (len(ordered) - max_items))

    if report.watchlist:
        console.rule()
        console.head("Watchlist")
        for entry in report.watchlist:
            print("  - %s" % entry)
    if report.risks:
        console.head("Risks / caveats")
        for entry in report.risks:
            print("  - %s" % entry)
    if report.failed_batches:
        console.warn(
            "%d of %d model answer(s) were not parsable JSON (see raw output)"
            % (report.failed_batches, report.batch_count)
        )
    if show_raw:
        console.rule()
        console.head("Raw model answers")
        for blob in report.raw_responses:
            print(blob)
            console.rule()


# -------------------------------------------------------------------- commands
async def _gather(settings: Settings, refs: Sequence[str], args: argparse.Namespace, store: NewsStore):
    """Connect, read every channel and update the incremental state file."""
    reader = TelegramNewsReader(settings)
    await reader.connect()
    try:
        state = None if getattr(args, "no_state", False) else store.state
        reads = await reader.fetch_many(
            refs,
            limit=args.limit,
            keywords=args.keyword,
            state=state,
            only_new=not getattr(args, "no_state", False),
        )
    finally:
        await reader.close()
    items: list[NewsItem] = [item for read in reads for item in read.items]
    items.sort(key=lambda item: (item.date, item.message_id), reverse=True)
    if state is not None:
        store.state.save()
    return items, reads


def cmd_doctor(args: argparse.Namespace) -> int:
    console.head("mcnews %s doctor" % __version__)
    problems: list[str] = []
    warnings: list[str] = []

    console.rule()
    console.field("python", "%s (%s)" % (sys.version.split()[0], sys.executable))
    if sys.version_info < (3, 9):
        problems.append("python 3.9 or newer is required")
    if TELEGRAM_IMPORT_ERROR is not None:
        console.err("telethon: NOT importable (%s)" % TELEGRAM_IMPORT_ERROR)
        problems.append("install dependencies: pip install -r requirements.txt")
    else:
        import telethon

        console.ok("telethon: installed (%s)" % getattr(telethon, "__version__", "?"))

    env_path = Path(args.env).expanduser() if getattr(args, "env", None) else PROJECT_ROOT / ".env"
    if env_path.is_file():
        console.ok(".env: %s" % env_path)
    else:
        console.warn(".env: not found at %s" % env_path)
        warnings.append("copy .env.example to .env and fill in your Telegram API keys")

    settings = resolve_settings(args)
    console.rule()
    for label, value in settings.describe():
        console.field(label, value)

    console.rule()
    telegram_problems = settings.telegram_problems()
    if telegram_problems:
        for problem in telegram_problems:
            console.err(problem)
            problems.append(problem)
        console.info("Create an app at https://my.telegram.org -> 'API development tools'.")
    else:
        console.ok("Telegram credentials present")
    if settings.session_exists():
        console.ok("session: %s" % settings.session_path.with_suffix(".session"))
    else:
        console.warn("session: not created yet")
        warnings.append("run: python main.py login")

    refs, ref_errors = parse_channel_refs(list(settings.channels))
    for bad in ref_errors:
        console.warn("bad channel ref in .env: %s" % bad)
        warnings.append("fix TELEGRAM_CHANNELS entry: %s" % bad)
    if refs:
        console.ok("channels from .env: %s" % ", ".join(refs))
    else:
        console.info("no channels configured in .env (pass -c @name per command)")

    console.rule()
    llm = build_llm(settings)
    healthy, message = llm.health()
    if healthy:
        console.ok("local model: %s" % message)
    else:
        console.err("local model: %s" % message)
        if settings.provider == "ollama":
            problems.append(
                "Ollama is not answering on %s - start it with 'ollama serve' and check 'ollama list'"
                % settings.resolved_base_url()
            )
        else:
            problems.append("local server is not answering on %s" % settings.resolved_base_url())

    console.rule()
    if problems:
        console.head("%d problem(s) to fix" % len(problems))
        for problem in problems:
            print("  - %s" % problem)
    else:
        console.ok("no blocking problems found")
    if warnings:
        console.head("%d note(s)" % len(warnings))
        for warning in warnings:
            print("  - %s" % warning)
    console.info("")
    console.info("next steps:  python main.py login   then   python main.py fetch -c @channelname")
    return EXIT_PROBLEM if problems else EXIT_OK


def cmd_login(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    problems = settings.telegram_problems()
    if problems:
        for problem in problems:
            console.err(problem)
        console.info("Get them at https://my.telegram.org -> 'API development tools', then put them in .env")
        return EXIT_PROBLEM

    console.info("Login is interactive: Telegram sends a code to your account.")
    console.info("Session file: %s" % settings.session_path.with_suffix(".session"))
    console.rule()

    async def run() -> None:
        reader = TelegramNewsReader(settings)
        await reader.connect(interactive=True)
        try:
            me = await reader.me()
            handle = "@%s" % me["username"] if me.get("username") else "no username"
            console.ok("logged in as %s (id=%s, %s)" % (me.get("name") or handle, me.get("id"), handle))
            dialogs = await reader.list_dialogs(limit=500)
            console.ok("%d channel(s)/group(s) visible to this account" % len(dialogs))
            for dialog in dialogs[:10]:
                print("  %s" % dialog.describe())
            if dialogs:
                console.info("try:  python main.py fetch -c %s -n 5" % dialogs[0].ref)
        finally:
            await reader.close()

    try:
        asyncio.run(run())
    except TelegramError as exc:
        console.err(str(exc))
        return EXIT_PROBLEM
    console.ok("session saved - later commands will not ask for a code again")
    return EXIT_OK


def cmd_channels(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)

    async def run() -> list[Any]:
        reader = TelegramNewsReader(settings)
        await reader.connect()
        try:
            return await reader.list_dialogs(limit=args.limit)
        finally:
            await reader.close()

    try:
        dialogs = asyncio.run(run())
    except TelegramError as exc:
        console.err(str(exc))
        return EXIT_PROBLEM

    if args.json:
        dump_json([dialog.__dict__ for dialog in dialogs])
        return EXIT_OK

    console.head("Channels and groups this account can read (%d)" % len(dialogs))
    console.rule()
    for dialog in dialogs:
        print(dialog.describe())
    console.rule()
    if dialogs:
        refs = [dialog.ref for dialog in dialogs[:5]]
        console.info("next:  python main.py fetch -c %s -n 5" % " -c ".join(refs))
    else:
        console.warn("nothing found: join a few public channels with this account first")
    return EXIT_OK


def cmd_fetch(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    refs = resolve_channel_refs(args, settings)
    store = NewsStore(settings.data_dir)
    console.info("channels : %s" % ", ".join(refs))
    console.info("reading up to %d post(s) per channel%s ..." % (
        args.limit,
        " (only unseen posts)" if not args.no_state else " (state ignored)",
    ))
    try:
        items, reads = asyncio.run(_gather(settings, refs, args, store))
    except TelegramNotConfigured as exc:
        console.err(str(exc))
        console.info("Create an app at https://my.telegram.org -> 'API development tools' and put the values in .env")
        return EXIT_PROBLEM
    except TelegramAuthRequired as exc:
        console.err(str(exc))
        return EXIT_PROBLEM

    print_reads(reads)
    ok_reads = [read for read in reads if read.ok]
    if not items and ok_reads:
        console.warn(
            "channels were reachable but nothing matched (state.json may already cover these posts - try --no-state)"
        )

    saved_path: Path | None = None
    if not args.no_save and items:
        saved_path, count = store.save_items(items)
        console.ok("saved %d post(s) -> %s" % (count, saved_path))

    if args.json:
        dump_json(
            {
                "channels": refs,
                "count": len(items),
                "saved": str(saved_path) if saved_path else None,
                "reads": [
                    {
                        "ref": read.ref,
                        "ok": read.ok,
                        "title": read.title,
                        "kind": read.kind,
                        "fetched": read.fetched,
                        "kept": read.kept,
                        "skipped": read.skipped,
                        "filtered": read.filtered,
                        "latest_id": read.last_message_id,
                        "error": read.error,
                    }
                    for read in reads
                ],
                "items": [item.to_dict() for item in items],
            }
        )
    else:
        print_items(items, show=args.show, header="Posts (newest first)")

    if not ok_reads:
        console.err("no channel could be read")
        return EXIT_PROBLEM
    console.ok("readable channels: %d/%d" % (len(ok_reads), len(reads)))
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    store = NewsStore(settings.data_dir)
    mode = "telegram"

    if args.from_raw:
        mode = "raw"
        path = None if args.from_raw == "latest" else Path(args.from_raw)
        items = store.load_items(path)
        source = path or store.latest_raw_path()
        if not items:
            console.err("no stored posts found in %s" % (source or (settings.data_dir / "raw")))
            console.info("run 'python main.py fetch -c @channelname' first, or use --from-raw <file.jsonl>")
            return EXIT_PROBLEM
        console.ok("loaded %d stored post(s) from %s" % (len(items), source))
    else:
        refs = resolve_channel_refs(args, settings)
        try:
            items, reads = asyncio.run(_gather(settings, refs, args, store))
        except (TelegramNotConfigured, TelegramAuthRequired) as exc:
            console.err(str(exc))
            return EXIT_PROBLEM
        print_reads(reads)
        if not items:
            console.warn("no new posts to analyse (try --no-state to ignore data/state.json)")
            return EXIT_OK
        if not args.no_save:
            saved_path, count = store.save_items(items)
            console.ok("saved %d raw post(s) -> %s" % (count, saved_path))

    analyzer = make_analyzer(args, settings)
    try:
        report = analyzer.analyze(items)
    except LLMUnavailable as exc:
        console.err("local model unreachable: %s" % exc)
        console.info("Start it (e.g. 'ollama serve') or point --base-url to your local server.")
        return EXIT_PROBLEM
    except LLMError as exc:
        console.err("model error: %s" % exc)
        return EXIT_PROBLEM

    context = {item.uid: item for item in items}
    if args.json:
        dump_json({"mode": mode, "report": report.to_dict(), "items": [item.to_dict() for item in items]})
    else:
        console.rule()
        console.head("Posts analysed (%d)" % len(items))
        for item in items:
            print("  %s | %s" % (console.style(item.label_line(), "dim"), shorten(item.text, 110)))
        print_report(report, context=context, show_raw=args.show_raw)

    if not args.no_save:
        report_path = store.save_report(report, tag=mode)
        console.ok("saved report -> %s" % report_path)
    return EXIT_OK


def cmd_watch(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    refs = resolve_channel_refs(args, settings)
    store = NewsStore(settings.data_dir)
    analyzer = make_analyzer(args, settings)
    interval = max(5, int(args.interval))

    console.info(
        "watching %s every %ds%s" % (", ".join(refs), interval, "" if args.iterations else " (Ctrl+C to stop)")
    )
    cycle = 0
    while True:
        cycle += 1
        started = datetime.now(timezone.utc)
        reads: list[Any] = []
        items: list[NewsItem] = []
        try:
            items, reads = asyncio.run(_gather(settings, refs, args, store))
        except (TelegramNotConfigured, TelegramAuthRequired) as exc:
            console.err(str(exc))
            return EXIT_PROBLEM
        except TelegramError as exc:
            console.err("poll failed: %s" % exc)

        console.rule()
        console.info(
            "[%s UTC] poll #%d -> %d new post(s)" % (started.strftime("%H:%M:%S"), cycle, len(items))
        )
        for read in reads:
            (console.ok if read.ok else console.err)(read.describe())

        if items:
            if not args.no_save:
                store.save_items(items)
            try:
                report = analyzer.analyze(items)
            except LLMError as exc:
                console.err("model error: %s" % exc)
                report = None
            if report is not None:
                if args.json:
                    dump_json(report.to_dict())
                else:
                    print_report(report, context={item.uid: item for item in items}, show_raw=args.show_raw)
                if not args.no_save:
                    console.ok("saved report -> %s" % store.save_report(report, tag="watch"))

        if args.iterations and cycle >= args.iterations:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    """Synthetic posts -> local model. Proves the AI half without Telegram."""
    settings = resolve_settings(args)
    items = build_sample_items()
    console.info("demo: %d synthetic post(s), no Telegram account involved" % len(items))
    analyzer = make_analyzer(args, settings)
    try:
        report = analyzer.analyze(items, note="synthetic demo data")
    except LLMUnavailable as exc:
        console.err("local model unreachable: %s" % exc)
        console.info("Start it (e.g. 'ollama serve') or point --base-url to your local server.")
        return EXIT_PROBLEM
    except LLMError as exc:
        console.err("model error: %s" % exc)
        return EXIT_PROBLEM

    if args.json:
        dump_json({"report": report.to_dict(), "items": [item.to_dict() for item in items]})
    else:
        print_items(items, show=0, header="Sample posts sent to the model")
        print_report(report, context={item.uid: item for item in items}, show_raw=args.show_raw)

    if not args.no_save:
        store = NewsStore(settings.data_dir)
        console.ok("saved report -> %s" % store.save_report(report, tag="demo"))
    return EXIT_OK


COMMANDS = {
    "doctor": cmd_doctor,
    "login": cmd_login,
    "channels": cmd_channels,
    "fetch": cmd_fetch,
    "analyze": cmd_analyze,
    "watch": cmd_watch,
    "demo": cmd_demo,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    console.init_stdout()
    console.enable(False if getattr(args, "no_color", False) else None)
    setup_logging(bool(getattr(args, "verbose", False)))

    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse already rejects this
        parser.error("unknown command: %s" % args.command)
        return EXIT_USAGE

    try:
        return handler(args)
    except UsageError as exc:
        console.err(str(exc))
        return EXIT_USAGE
    except KeyboardInterrupt:
        console.info("")
        console.info("interrupted - bye")
        return EXIT_OK
    except TelegramError as exc:
        console.err(str(exc))
        return EXIT_PROBLEM
    except LLMError as exc:
        console.err(str(exc))
        return EXIT_PROBLEM







