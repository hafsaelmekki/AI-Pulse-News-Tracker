from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_pulse_tracker.models import UpsertResult


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
    parser.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp to refetch articles published after this moment (e.g. 2024-04-01T12:30:00Z).",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore the incremental cursor and refetch the latest batch even if already ingested.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ensure_upsert_result(result: UpsertResult | list[str]) -> UpsertResult:
    if isinstance(result, UpsertResult):
        return result
    return UpsertResult(ids=list(result), created=len(result), updated=0)


def run_once(pipeline: NewsAnalyzerPipeline, args: argparse.Namespace) -> None:
    run_kwargs = _build_run_kwargs(args)
    result = _ensure_upsert_result(pipeline.run(query=args.query, **run_kwargs))
    print(
        f"[{_timestamp()}] {result.created} new / {result.updated} refreshed "
        f"(total {len(result.ids)})"
    )


def _build_run_kwargs(args: argparse.Namespace) -> dict:
    try:
        since = _parse_since(args.since) if args.since else None
    except ValueError as exc:
        print(f"Invalid --since value: {exc}", file=sys.stderr)
        sys.exit(2)
    incremental = not args.full_refresh and not since
    return {"after": since, "incremental": incremental}


def run_continuous(pipeline: NewsAnalyzerPipeline, args: argparse.Namespace) -> None:
    delay = max(args.interval, 30)
    print(
        f"[{_timestamp()}] Starting real-time tracking loop every {delay} seconds "
        f"(press Ctrl+C to stop)."
    )
    try:
        while True:
            run_once(pipeline, args)
            time.sleep(delay)
    except KeyboardInterrupt:
        print(f"[{_timestamp()}] Stopped real-time tracking.")


def main() -> None:
    from ai_pulse_tracker.pipeline import NewsAnalyzerPipeline

    args = parse_args()
    pipeline = NewsAnalyzerPipeline()
    if args.interval and args.interval > 0:
        run_continuous(pipeline, args)
    else:
        run_once(pipeline, args)


def _parse_since(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


if __name__ == "__main__":
    main()
