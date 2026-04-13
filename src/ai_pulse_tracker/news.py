from __future__ import annotations

import logging
from datetime import datetime
import requests

from .config import Settings
from .models import Article

LOGGER = logging.getLogger(__name__)


class NewsClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session or requests.Session()

    def fetch_articles(self, query: str | None = None) -> list[Article]:
        params = {
            "q": query or self._settings.news_query,
            "language": self._settings.news_language,
            "pageSize": self._settings.news_batch_size,
            "apiKey": self._settings.news_api_key,
            "sortBy": "publishedAt",
        }
        response = self._session.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            raise RuntimeError(f"NewsAPI responded with: {payload}")

        articles: list[Article] = []
        for entry in payload.get("articles", []):
            published_raw = entry.get("publishedAt") or datetime.utcnow().isoformat()
            published_at = _parse_date(published_raw)
            description = entry.get("description") or ""

            articles.append(
                Article(
                    source=entry["source"].get("name", "Unknown"),
                    title=entry.get("title", "Untitled"),
                    description=description,
                    url=entry.get("url", ""),
                    published_at=published_at,
                )
            )

        LOGGER.info("Fetched %s articles", len(articles))
        return articles


def _parse_date(value: str) -> datetime:
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        LOGGER.warning("Unable to parse date '%s', defaulting to now", value)
        return datetime.utcnow()
