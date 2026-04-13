from __future__ import annotations

from datetime import datetime, timezone

from ai_pulse_tracker.news import _format_from_param
from news_analyzer import _parse_since


def test_format_from_param_converts_to_utc_and_increments():
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _format_from_param(ts) == "2024-01-01T12:00:01Z"


def test_format_from_param_handles_naive_datetime():
    ts = datetime(2024, 1, 1, 12, 0, 0)
    assert _format_from_param(ts) == "2024-01-01T12:00:01Z"


def test_parse_since_supports_z_suffix():
    dt = _parse_since("2024-02-01T10:30:00Z")
    assert dt.isoformat() == "2024-02-01T10:30:00+00:00"


def test_parse_since_assumes_utc_for_naive():
    dt = _parse_since("2024-02-01T10:30:00")
    assert dt.isoformat() == "2024-02-01T10:30:00+00:00"
