from __future__ import annotations

import logging
from typing import Iterable

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from .config import Settings
from .enrichment import (
    compute_importance_score,
    extract_keywords,
    generate_article_summary,
)
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

        analyzed: list[AnalyzedArticle] = []
        batch_size = 10  # Azure Text Analytics sentiment endpoint max documents per request
        for start in range(0, len(article_list), batch_size):
            batch = article_list[start : start + batch_size]
            documents = [
                f"{article.title}. {article.description}".strip()
                for article in batch
            ]

            try:
                responses = self._client.analyze_sentiment(documents=documents)
            except HttpResponseError as exc:
                raise RuntimeError("Azure AI sentiment analysis failed") from exc

            for article, result in zip(batch, responses):
                keywords = extract_keywords(article.title, article.description)
                confidence_scores = result.confidence_scores
                importance_score = compute_importance_score(
                    title=article.title,
                    description=article.description,
                    url=article.url,
                    source=article.source,
                    sentiment_confidence=max(
                        confidence_scores.positive,
                        confidence_scores.neutral,
                        confidence_scores.negative,
                    ),
                    keywords=keywords,
                )
                summary = generate_article_summary(
                    title=article.title,
                    description=article.description,
                    keywords=keywords,
                    sentiment=result.sentiment,
                    importance_score=importance_score,
                )
                analyzed.append(
                    AnalyzedArticle(
                        source=article.source,
                        title=article.title,
                        description=article.description,
                        url=article.url,
                        published_at=article.published_at,
                        sentiment=result.sentiment,
                        confidence_pos=confidence_scores.positive,
                        confidence_neu=confidence_scores.neutral,
                        confidence_neg=confidence_scores.negative,
                        keywords=keywords,
                        importance_score=importance_score,
                        summary=summary,
                    )
                )

        LOGGER.info("Analyzed %s articles", len(analyzed))
        return analyzed
