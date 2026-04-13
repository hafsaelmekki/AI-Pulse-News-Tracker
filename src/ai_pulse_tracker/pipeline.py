from __future__ import annotations

import logging

from .config import Settings, load_settings
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

    def run(self, query: str | None = None) -> list[str]:
        configure_logging()
        articles = self._news_client.fetch_articles(query)
        if not articles:
            LOGGER.warning("No new articles retrieved from NewsAPI")
            return []
        analyzed = self._sentiment_client.analyze(articles)
        return self._repository.upsert_articles(analyzed)

    def load_dashboard_rows(self) -> list[dict]:
        return self._repository.load_all()
