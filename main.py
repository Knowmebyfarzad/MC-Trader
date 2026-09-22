#!/usr/bin/env python
"""McNews entry point.

    python main.py doctor
    python main.py login
    python main.py channels
    python main.py fetch -c @channelname -n 10
    python main.py analyze -c @channelname -n 10
    python main.py watch -c @channelname --interval 60
    python main.py demo

Run ``python main.py -h`` for the full list of options.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running "python main.py" from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcnews.cli import main  # noqa: E402  (import after sys.path setup)

if __name__ == "__main__":
    raise SystemExit(main())
