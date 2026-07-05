from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .trends import normalize_keywords


AI_RELEVANCE_KEYWORDS = (
    "artificial intelligence",
    "intelligence artificielle",
    "generative ai",
    "genai",
    "chatgpt",
    "openai",
    "anthropic",
    "claude",
    "gemini",
    "mistral",
    "llm",
    "rag",
    "ai agent",
    "ai agents",
    "agent ia",
    "machine learning",
    "deep learning",
    "copilot",
    "nvidia",
    "azure ai",
    "automation",
    "automatisation",
    "ai",
    "ia",
)


def ai_relevance_reason(article: Mapping[str, Any] | object) -> str:
    text = _article_relevance_text(article)
    if not text:
        return ""

    normalized = text.lower()
    for keyword in AI_RELEVANCE_KEYWORDS:
        if _matches_keyword(normalized, keyword):
            return f"matched keyword: {keyword}"
    return ""


def is_ai_related(article: Mapping[str, Any] | object) -> bool:
    return bool(ai_relevance_reason(article))


def _article_relevance_text(article: Mapping[str, Any] | object) -> str:
    values = [
        _get_article_value(article, "title"),
        _get_article_value(article, "description"),
        _get_article_value(article, "content"),
        _get_article_value(article, "summary"),
    ]
    keywords = _get_article_value(article, "keywords")
    if keywords:
        values.append(" ".join(normalize_keywords(keywords)))
    return " ".join(str(value or "") for value in values)


def _get_article_value(article: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(article, Mapping):
        return article.get(key)
    return getattr(article, key, None)


def _matches_keyword(text: str, keyword: str) -> bool:
    if keyword in {"ai", "ia", "llm", "rag"}:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text))
    return keyword in text
