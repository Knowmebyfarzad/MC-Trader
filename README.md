# McNews (beta) - Telegram news reader + local AI analysis

An MVP that answers one question first: **can this machine read the news from
Telegram channels, and can the local AI model make sense of it?**

It does exactly two things:

1. **READS** posts from public Telegram channels (or private ones the account
   already joined) by username, link or numeric id - read only, it never sends,
   joins or reacts to anything.
2. **ANALYSES** those posts with a **local** model (Ollama, or any
   OpenAI-compatible local server) and prints/saves a structured report:
   sentiment, market impact 1-10, affected assets, topics, watchlist, risks.


# Preview 
![home]("F:\MC Trader\Firstlook\home.png")


No cloud services, no API keys for the AI, one Python dependency (`telethon`).

```
Telegram channels ──► Telethon reader ──► NewsItem list ──► batched prompts ──► local model
                                              │                                     │
                                              ▼                                     ▼
                                     data/raw/*.jsonl                    data/analyses/*.json
```

## Files

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry point (`doctor`, `login`, `channels`, `fetch`, `analyze`, `watch`, `demo`) |
| `webserver.py` | Local web UI entry point (browser frontend for fetch/analyze) |
| `web/index.html` | The web page served by `webserver.py` (no framework, no build step) |
| `mcnews/config.py` | `.env` loader + `Settings` (no `python-dotenv` needed) |
| `mcnews/telegram_source.py` | Telethon reader: login, resolve refs, read posts, list dialogs |
| `mcnews/llm.py` | Local model clients (Ollama `/api/chat`, OpenAI-compatible `/chat/completions`) + JSON extractor |
| `mcnews/prompts.py` | System prompt and the strict JSON contract |
| `mcnews/analyzer.py` | Batching, output normalisation, merge of multi-batch answers |
| `mcnews/store.py` | `data/raw/*.jsonl`, `data/analyses/*.json`, `data/state.json` (incremental reads) |
| `mcnews/models.py` | `NewsItem` / `AnalysisReport` dataclasses |
| `mcnews/cli.py` | Commands, output formatting, exit codes |
| `mcnews/demo_data.py` | Synthetic posts for the `demo` command |
| `tests/` | 146 offline tests (stdlib `unittest`, fake Telegram client + stub HTTP server) |

## Requirements

* Windows / macOS / Linux, **Python 3.9+** (validated here on Python 3.14.7)
* `pip install -r requirements.txt` -> `telethon` (validated with 1.45.0)
* A local model server:

  ```powershell
  ollama serve            # keep running in its own window
  ollama pull qwen2.5:3b  # small and fast; qwen3:4b / llama3.1:8b also work
  ```

  LM Studio / llama.cpp / vLLM work too: set `LLM_PROVIDER=openai` and
  `LLM_BASE_URL=http://127.0.0.1:1234/v1`.

## Setup (5 minutes)

### 1. Telegram API credentials (required once)

1. Open <https://my.telegram.org> and log in with your phone number.
2. Go to **API development tools**, create an app (any name/URL is fine).
3. Copy `api_id` and `api_hash`.

```powershell
copy .env.example .env
notepad .env      # fill TELEGRAM_API_ID and TELEGRAM_API_HASH
```

Optional but handy: `TELEGRAM_CHANNELS=@channel_one,@channel_two` so you can run
commands without `-c` every time.

### 2. Check everything

```powershell
python main.py doctor
```

It verifies Python, telethon, `.env`, credentials, the session file, the channel
references and the local model, then tells you exactly what is missing.

### 3. Log in (interactive, once)

```powershell
python main.py login
```

Telethon asks for your phone, the login code Telegram sends you and your 2FA
password if you have one. The session is stored in `sessions/mcnews.session` -
treat that file like a password, never commit or share it (it is git-ignored).

### 4. Prove the AI half works (no Telegram needed)

```powershell
python main.py demo
```

Runs the analysis on 5 synthetic posts and prints the report. This tells you the
model + prompt + parsing chain works before touching Telegram.

## Usage

```powershell
# which channels can this account read? (copy the refs from here)
python main.py channels

# BETA TEST: can we read news from channels? (no AI involved)
python main.py fetch -c "@channelname" -n 10

# read + analyse with the local model
python main.py analyze -c "@channelname" -n 15

# several channels, keyword filter, extra focus for the analyst
python main.py fetch -c "@macro_news" -c "https://t.me/gold_wire" -k gold -k dollar -n 30
python main.py analyze -c "@macro_news" --focus "gold, USD/IRR, crypto" --show-raw

# re-analyse posts already collected (no Telegram call)
python main.py analyze --from-raw

# live polling: analyse only what is new
python main.py watch -c "@channelname" --interval 60
```

> **PowerShell note:** always quote a channel reference. PowerShell silently
> treats a bare `@word` as a splat expression and drops the argument entirely
> (`cmd`-style `-c @name` works, but `-c "@name"` is the safe form everywhere).
> If the value does get dropped you now get a clear message instead of an
> argparse usage dump.

| Command | What it does | Needs Telegram? | Needs the model? |
| --- | --- | --- | --- |
| `doctor` | environment / config / model check | no | only for its health check |
| `login` | create the session (interactive) | yes | no |
| `channels` | list readable channels and groups | yes | no |
| `fetch` | read posts, print + save them | yes | no |
| `analyze` | read (or load) posts, then analyse | optional (`--from-raw`) | yes |
| `watch` | poll and analyse new posts | yes | yes |
| `demo` | synthetic posts -> local model | no | yes |

Useful flags: `-n/--limit`, `-k/--keyword`, `--no-state` (re-read the newest
posts even if already seen), `--json` (machine readable), `--no-save`,
`--show-raw` (dump raw model answers), `--model`, `--provider`, `--base-url`,
`--focus`, `--verbose`, `--no-color`. Exit codes: `0` ok, `1` problem,
`2` usage/configuration error.

### Channel reference formats

```
@channelname
https://t.me/channelname
https://t.me/s/channelname
https://t.me/c/1234567890/42      (private channel this account already joined)
-1001234567890                    (numeric id)
```

Invite links (`t.me/+hash`, `joinchat`) are recognised but **not joined** - this
MVP never joins groups; join manually if you want to read one.

## What gets saved

| Path | Content |
| --- | --- |
| `data/raw/raw-YYYYMMDD.jsonl` | every post read (one JSON object per line) |
| `data/analyses/analysis-<timestamp>-<mode>.json` | full report, incl. raw model answers |
| `data/state.json` | newest message id per channel, so `watch`/`analyze` only fetch what is new |
| `sessions/mcnews.session` | Telegram login session (**secret**, git-ignored) |

## What the model must return

The prompt asks for exactly this JSON (and nothing else). Anything that is not
parsable JSON is kept as a raw answer and reported instead of crashing:

```json
{
  "overall": {
    "summary": "2-4 sentence digest",
    "sentiment": "bullish | bearish | neutral | mixed",
    "confidence": 0.0,
    "market_impact": 1,
    "tradable": true,
    "time_horizon": "intraday | days | weeks | months | unspecified",
    "key_drivers": ["short phrase"]
  },
  "items": [
    {
      "channel": "-1001234567890",
      "message_id": 4821,
      "summary": "one sentence",
      "sentiment": "bullish",
      "impact": 5,
      "assets": ["gold"],
      "topics": ["macro"],
      "actionable": false
    }
  ],
  "watchlist": ["what to keep monitoring"],
  "risks": ["what could make this reading wrong"]
}
```

Multi-batch runs are merged locally (no extra model call): sentiment by
majority (tie -> `mixed`), `market_impact` = max, `tradable` = any, confidence =
average, summaries joined with ` | `.

## How failures behave

| Situation | Behaviour |
| --- | --- |
| no `TELEGRAM_API_ID/HASH` | clear message, exit 1, tells you to use my.telegram.org |
| session not created | "run: python main.py login", exit 1 |
| channel private / wrong username | that channel is marked ERROR, other channels still run |
| Telegram rate limit | reported as flood-wait; lower `-n`, retry later |
| empty / media-only post | skipped and counted (`skipped=`), not sent to the model |
| model server down | `fetch` still works; `analyze` stops with a hint to start it |
| model answers with non-JSON | batch marked failed, raw answer stored, report still produced |
| session revoked by Telegram | delete the `.session` file and run `login` again |

## Web UI (browser frontend)

Prefer clicking to typing? The same `fetch` / `analyze` features are available
as a small local website:

```powershell
python webserver.py                  # http://127.0.0.1:8000
python webserver.py --port 9000      # custom port
```

Open the address in a browser, type a channel name (step 1), pick how many
posts to read (step 2), optionally add keywords / an analysis focus, then press
**Read posts** or **Read & Analyze**. The page shows live status pills
(Telegram session, AI model), per-channel read results, the analysis report
(sentiment, market impact, key drivers, watchlist, risks) and the posts
themselves.

To share it while your computer is up, expose it through a Cloudflare quick
tunnel:

```powershell
python webserver.py --host 0.0.0.0
cloudflared tunnel --url http://localhost:8000
```

One run executes at a time (the Telegram session file is not concurrent-safe);
a second visitor gets a friendly "another run is in progress" message.

## Tests

The whole suite is offline: the Telegram client is faked and the model server is
a local stub HTTP server, so no credentials, no network and no running Ollama are
needed.

```powershell
python -m unittest discover -s tests -v     # 146 tests
python -m unittest tests.test_analyzer -v   # one module
```

Covered: `.env`/`Settings` parsing, text cleaning/chunking/coercion, `NewsItem`
round-trips, channel reference parsing, the reader (limits, `min_id` state,
keyword filter, skipped empty posts, error/flood handling, dialog listing),
JSON extraction, both model clients (request payloads, health, error mapping),
batching + merge + normalisation, and the JSONL/report stores.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `TELEGRAM_API_ID is not set` | create the app at my.telegram.org and fill `.env` |
| `this session is not logged in yet` | `python main.py login` |
| `cannot resolve @name` | check the spelling; private channels only work if the account already joined them |
| `flood-wait 120s` | wait, lower `-n`, or increase `FLOOD_SLEEP` |
| `Ollama is not answering on http://127.0.0.1:11434` | start `ollama serve`, or set `LLM_BASE_URL` |
| model is very slow | use a smaller model (`qwen2.5:3b`, `qwen3:1.7b`) and reduce `-n`/`BATCH_CHARS` |
| report has no `items` | the model chose not to score them; re-run with `--show-raw` to see its raw answer |
| `no channels given` in PowerShell | quote the ref: `-c "@name"` (a bare `@word` is swallowed by the shell) |
| Windows console shows `?` characters | `python main.py ... | Out-File -Encoding utf8 file.txt` or run in Windows Terminal |

## Limitations (this is a beta)

* Read-only by design: no sending, no joining, no reactions - by construction.
* Public channels and channels the account already joined; invite links are not
  auto-joined.
* Media-only posts (no caption) are counted as skipped, never analysed.
* No OCR, no image analysis, no scheduling/daemon mode beyond `watch`.
* `data/state.json` tracks the newest message id per channel; a channel addressed
  sometimes by `@name` and sometimes by numeric id is tracked as two entries.
* Local 3B-class models hallucinate: treat the output as a first draft, always
  verify against the source post (the report prints the original link).
* Analysis quality depends on the prompt and the model - this is a plumbing MVP,
  not a trading signal generator. **Not financial advice.**

## Legality / etiquette

Reading Telegram channels through your own account is subject to Telegram's
Terms of Service; prefer public channels, keep `--limit` modest, respect
flood-waits, and do not republish copyrighted content. The session file grants
full access to your account - keep it private.

