from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .trends import normalize_keywords


def answer_question(question: str, retrieved_articles: list[Mapping[str, Any]]) -> str:
    question = question.strip()
    if not retrieved_articles:
        return (
            f"I could not find stored articles that answer: {question}. "
            "Try a broader question or ingest more recent articles."
        )

    citations = _citation_labels(retrieved_articles)
    keywords = _top_keywords(retrieved_articles)
    sentiment_counts = Counter(
        str(article.get("sentiment", "")).strip().lower()
        for article in retrieved_articles
        if str(article.get("sentiment", "")).strip()
    )
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "unknown"

    lines = [
        f"Answer based on {len(retrieved_articles)} retrieved article(s):",
        "",
        _overview_sentence(question, keywords, dominant_sentiment, citations),
        "",
        "Key signals:",
    ]
    lines.extend(_signal_lines(retrieved_articles, citations))
    lines.extend(["", "Sources:"])
    lines.extend(_source_lines(retrieved_articles, citations))
    return "\n".join(lines)


def _overview_sentence(
    question: str,
    keywords: list[str],
    dominant_sentiment: str,
    citations: list[str],
) -> str:
    topic_text = ", ".join(keywords[:5]) if keywords else "the retrieved articles"
    cited = " ".join(citations[:3])
    return (
        f"For '{question}', the strongest signals are {topic_text}. "
        f"The dominant sentiment in the retrieved set is {dominant_sentiment}. {cited}"
    ).strip()


def _signal_lines(
    retrieved_articles: list[Mapping[str, Any]],
    citations: list[str],
) -> list[str]:
    lines: list[str] = []
    for article, citation in zip(retrieved_articles[:3], citations):
        title = str(article.get("title", "")).strip() or "Untitled article"
        summary = str(article.get("summary", "")).strip()
        matched_terms = str(article.get("matched_terms", "")).strip()
        signal = summary or title
        suffix = f" Matched terms: {matched_terms}." if matched_terms else ""
        lines.append(f"- {signal}{suffix} {citation}")
    return lines


def _source_lines(
    retrieved_articles: list[Mapping[str, Any]],
    citations: list[str],
) -> list[str]:
    lines: list[str] = []
    for article, citation in zip(retrieved_articles, citations):
        source = str(article.get("source", "")).strip() or "Unknown source"
        title = str(article.get("title", "")).strip() or "Untitled article"
        url = str(article.get("url", "")).strip()
        if url:
            lines.append(f"- {citation} {source}: {title} ({url})")
        else:
            lines.append(f"- {citation} {source}: {title}")
    return lines


def _top_keywords(retrieved_articles: list[Mapping[str, Any]], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for article in retrieved_articles:
        counter.update(normalize_keywords(article.get("keywords")))
    return [keyword for keyword, _ in counter.most_common(limit)]


def _citation_labels(retrieved_articles: list[Mapping[str, Any]]) -> list[str]:
    return [f"[{index}]" for index in range(1, len(retrieved_articles) + 1)]
