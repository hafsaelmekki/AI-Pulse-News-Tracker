from __future__ import annotations

import logging
from typing import Iterable

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from .config import Settings
from .models import AnalyzedArticle, Article

LOGGER = logging.getLogger(__name__)


class SentimentClient:
    def __init__(self, settings: Settings) -> None:
        credential = AzureKeyCredential(settings.azure_ai_key)
        self._client = TextAnalyticsClient(settings.azure_ai_endpoint, credential)

    def analyze(self, articles: Iterable[Article]) -> list[AnalyzedArticle]:
        article_list = list(articles)
        if not article_list:
            return []
        documents = [f"{article.title}. {article.description}".strip() for article in article_list]

        try:
            responses = self._client.analyze_sentiment(documents=documents)
        except HttpResponseError as exc:
            raise RuntimeError("Azure AI sentiment analysis failed") from exc

        analyzed: list[AnalyzedArticle] = []
        for article, result in zip(article_list, responses):
            analyzed.append(
                AnalyzedArticle(
                    source=article.source,
                    title=article.title,
                    description=article.description,
                    url=article.url,
                    published_at=article.published_at,
                    sentiment=result.sentiment,
                    confidence_pos=result.confidence_scores.positive,
                    confidence_neu=result.confidence_scores.neutral,
                    confidence_neg=result.confidence_scores.negative,
                )
            )

        LOGGER.info("Analyzed %s articles", len(analyzed))
        return analyzed
