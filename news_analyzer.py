from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_pulse_tracker.pipeline import NewsAnalyzerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch latest AI articles, analyze sentiment, and persist to Cosmos DB.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Override the default NewsAPI query term.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Enable real-time tracking by running every N seconds (min 30). 0 runs once.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_once(pipeline: NewsAnalyzerPipeline, query: str | None) -> None:
    inserted_ids = pipeline.run(query=query)
    print(f"[{_timestamp()}] Persisted {len(inserted_ids)} analyzed articles.")


def run_continuous(pipeline: NewsAnalyzerPipeline, query: str | None, interval: int) -> None:
    delay = max(interval, 30)
    print(
        f"[{_timestamp()}] Starting real-time tracking loop every {delay} seconds "
        f"(press Ctrl+C to stop)."
    )
    try:
        while True:
            run_once(pipeline, query)
            time.sleep(delay)
    except KeyboardInterrupt:
        print(f"[{_timestamp()}] Stopped real-time tracking.")


def main() -> None:
    args = parse_args()
    pipeline = NewsAnalyzerPipeline()
    if args.interval and args.interval > 0:
        run_continuous(pipeline, args.query, args.interval)
    else:
        run_once(pipeline, args.query)


if __name__ == "__main__":
    main()
