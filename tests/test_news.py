from __future__ import annotations

from datetime import datetime, timezone

from ai_pulse_tracker.config import Settings
from ai_pulse_tracker.news import (
    NewsClient,
    _clamp_after_to_lookback,
    _format_from_param,
    _format_to_param,
)
from news_analyzer import _parse_since, _parse_until


def test_format_from_param_converts_to_utc_and_increments():
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _format_from_param(ts) == "2024-01-01T12:00:01Z"


def test_format_from_param_handles_naive_datetime():
    ts = datetime(2024, 1, 1, 12, 0, 0)
    assert _format_from_param(ts) == "2024-01-01T12:00:01Z"


def test_format_to_param_does_not_increment():
    ts = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert _format_to_param(ts) == "2024-12-31T23:59:59Z"


def test_parse_since_supports_z_suffix():
    dt = _parse_since("2024-02-01T10:30:00Z")
    assert dt.isoformat() == "2024-02-01T10:30:00+00:00"


def test_parse_since_assumes_utc_for_naive():
    dt = _parse_since("2024-02-01T10:30:00")
    assert dt.isoformat() == "2024-02-01T10:30:00+00:00"


def test_parse_until_supports_z_suffix():
    dt = _parse_until("2024-12-31T23:59:59Z")
    assert dt.isoformat() == "2024-12-31T23:59:59+00:00"


def test_clamp_after_to_lookback_limits_old_dates():
    now = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
    old_date = datetime(2026, 4, 12, 14, 31, 0, tzinfo=timezone.utc)

    clamped = _clamp_after_to_lookback(old_date, 29, now=now)

    assert clamped == datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_clamp_after_to_lookback_can_be_disabled():
    old_date = datetime(2026, 4, 12, 14, 31, 0, tzinfo=timezone.utc)

    assert _clamp_after_to_lookback(old_date, 0) == old_date


class _FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "AI News"},
                    "title": "OpenAI launches new AI agent tools",
                    "description": "Enterprise automation with generative AI.",
                    "content": "",
                    "url": "https://example.com/ai",
                    "publishedAt": "2026-07-01T10:00:00Z",
                },
                {
                    "source": {"name": "Travel News"},
                    "title": "Hotel summer vacation deals",
                    "description": "Local tourism offer for families.",
                    "content": "",
                    "url": "https://example.com/hotel",
                    "publishedAt": "2026-07-01T11:00:00Z",
                },
            ],
        }


class _FakeSession:
    def get(self, *args, **kwargs) -> _FakeResponse:
        return _FakeResponse()


def test_fetch_articles_filters_non_ai_articles():
    settings = Settings(
        azure_ai_endpoint="https://azure.example",
        azure_ai_key="azure-key",
        news_api_key="news-key",
        cosmos_endpoint="https://cosmos.example",
        cosmos_key="cosmos-key",
        news_batch_size=10,
    )

    articles = NewsClient(settings, session=_FakeSession()).fetch_articles()

    assert len(articles) == 1
    assert articles[0].title == "OpenAI launches new AI agent tools"
    assert articles[0].ai_relevance is True
    assert articles[0].ai_relevance_reason
