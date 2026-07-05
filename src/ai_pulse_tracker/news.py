from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
import math
import requests

from .config import Settings
from .models import Article

LOGGER = logging.getLogger(__name__)
MAX_NEWSAPI_PAGES = 5


class NewsAPIError(RuntimeError):
    """Raised when NewsAPI rejects a request."""


class NewsClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()

    def fetch_articles(
        self,
        query: str | None = None,
        after: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Article]:
        desired_count = max(1, self._settings.news_batch_size)
        page_size = min(desired_count, 100)
        max_pages = min(MAX_NEWSAPI_PAGES, math.ceil(desired_count / page_size))

        effective_after = _clamp_after_to_lookback(
            after,
            self._settings.news_max_lookback_days,
        )
        if after and effective_after and effective_after > after:
            LOGGER.info(
                "Clamped NewsAPI from date from %s to %s to respect lookback limit",
                after.isoformat(),
                effective_after.isoformat(),
            )

        articles: list[Article] = []
        for page in range(1, max_pages + 1):
            params = {
                "q": query or self._settings.news_query,
                "language": self._settings.news_language,
                "pageSize": page_size,
                "page": page,
                "apiKey": self._settings.news_api_key,
                "sortBy": "publishedAt",
            }
            if effective_after:
                params["from"] = _format_from_param(effective_after)
            if until:
                params["to"] = _format_to_param(until)
            response = self._session.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=15,
            )
            _raise_for_newsapi_error(response)
            payload = response.json()

            if payload.get("status") != "ok":
                raise RuntimeError(f"NewsAPI responded with: {payload}")

            batch = []
            for entry in payload.get("articles", []):
                published_raw = entry.get("publishedAt") or datetime.utcnow().isoformat()
                published_at = _parse_date(published_raw)
                description = entry.get("description") or ""

                batch.append(
                    Article(
                        source=entry["source"].get("name", "Unknown"),
                        title=entry.get("title", "Untitled"),
                        description=description,
                        url=entry.get("url", ""),
                        published_at=published_at,
                    )
                )

            articles.extend(batch)
            LOGGER.debug("Fetched %s articles on page %s", len(batch), page)

            if len(articles) >= desired_count or len(batch) < page_size:
                break

        LOGGER.info("Fetched %s articles", min(len(articles), desired_count))
        return articles[:desired_count]


def _parse_date(value: str) -> datetime:
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        LOGGER.warning("Unable to parse date '%s', defaulting to now", value)
        return datetime.utcnow()


def _format_from_param(value: datetime) -> str:
    return _format_newsapi_datetime(value, increment_seconds=1)


def _format_to_param(value: datetime) -> str:
    return _format_newsapi_datetime(value)


def _format_newsapi_datetime(value: datetime, increment_seconds: int = 0) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc) + timedelta(seconds=increment_seconds)
    value = value.replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _clamp_after_to_lookback(
    value: datetime | None,
    max_lookback_days: int,
    *,
    now: datetime | None = None,
) -> datetime | None:
    if value is None or max_lookback_days <= 0:
        return value
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    cutoff = now.astimezone(timezone.utc) - timedelta(days=max_lookback_days)
    return max(value.astimezone(timezone.utc), cutoff)


def _raise_for_newsapi_error(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = _extract_newsapi_error_message(response)
        status_code = response.status_code
        if status_code == 426:
            message = (
                message
                or "NewsAPI requires an upgraded plan for the requested date range."
            )
            message = (
                f"{message} Reduce NEWS_MAX_LOOKBACK_DAYS or use a newer date range."
            )
        raise NewsAPIError(
            f"NewsAPI request failed with status {status_code}: {message}"
        ) from exc


def _extract_newsapi_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    message = payload.get("message") if isinstance(payload, dict) else None
    return str(message or "").strip()
