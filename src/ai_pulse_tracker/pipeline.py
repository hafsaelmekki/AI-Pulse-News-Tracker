from __future__ import annotations

import logging

from datetime import datetime

from .config import Settings, load_settings
from .models import UpsertResult
from .news import NewsClient
from .sentiment import SentimentClient
from .storage import CosmosRepository

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class NewsAnalyzerPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._news_client = NewsClient(self._settings)
        self._sentiment_client = SentimentClient(self._settings)
        self._repository = CosmosRepository(self._settings)

    def run(
        self,
        query: str | None = None,
        *,
        after: datetime | None = None,
        incremental: bool = True,
    ) -> UpsertResult:
        configure_logging()
        effective_after = after
        if incremental:
            if effective_after is None:
                effective_after = self._repository.latest_published_at()
        articles = self._news_client.fetch_articles(query, after=effective_after)
        if not articles:
            LOGGER.warning("No new articles retrieved from NewsAPI")
            return UpsertResult(ids=[], created=0, updated=0)
        analyzed = self._sentiment_client.analyze(articles)
        return self._repository.upsert_articles(analyzed)

    def load_dashboard_rows(self) -> list[dict]:
        return self._repository.load_all()
