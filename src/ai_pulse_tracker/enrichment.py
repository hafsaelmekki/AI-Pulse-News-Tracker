from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9+'-]*")

_STOPWORDS = {
    "about",
    "after",
    "and",
    "avec",
    "dans",
    "des",
    "elle",
    "elles",
    "for",
    "from",
    "have",
    "les",
    "leur",
    "leurs",
    "mais",
    "more",
    "nous",
    "not",
    "our",
    "par",
    "pas",
    "plus",
    "pour",
    "que",
    "qui",
    "ses",
    "son",
    "sur",
    "the",
    "this",
    "une",
    "vous",
    "with",
    "your",
}

_SHORT_KEYWORDS = {"ai", "ia", "ml"}


def extract_keywords(
    title: str,
    description: str | None = None,
    *,
    limit: int = 8,
) -> list[str]:
    text = f"{title or ''} {description or ''}"
    words = [_normalize_token(token) for token in _TOKEN_RE.findall(text)]
    candidates = [word for word in words if _is_keyword_candidate(word)]
    if not candidates:
        return []

    counts = Counter(candidates)
    first_seen: dict[str, int] = {}
    for index, word in enumerate(candidates):
        first_seen.setdefault(word, index)
    ranked = sorted(
        counts,
        key=lambda word: (-counts[word], first_seen[word], word),
    )
    return ranked[: max(1, limit)]


def compute_importance_score(
    *,
    title: str,
    description: str | None,
    url: str,
    source: str,
    sentiment_confidence: float,
    keywords: Iterable[str],
) -> float:
    keyword_count = len(list(keywords))
    completeness_score = 0.0
    if title.strip():
        completeness_score += 15.0
    if description and description.strip():
        completeness_score += 15.0
    if url.strip():
        completeness_score += 10.0
    if source.strip():
        completeness_score += 10.0

    confidence_score = max(0.0, min(sentiment_confidence, 1.0)) * 30.0
    keyword_score = min(keyword_count, 8) / 8 * 20.0
    return round(completeness_score + confidence_score + keyword_score, 2)


def generate_article_summary(
    *,
    title: str,
    description: str | None,
    keywords: Iterable[str],
    sentiment: str,
    importance_score: float,
    max_length: int = 320,
) -> str:
    title_text = _clean_text(title)
    description_text = _clean_text(description or "")
    if title_text and description_text and description_text.lower() not in title_text.lower():
        summary = f"{title_text} - {description_text}"
    else:
        summary = description_text or title_text

    topics = ", ".join(list(keywords)[:3])
    signals: list[str] = []
    if topics:
        signals.append(f"Topics: {topics}")
    if sentiment:
        signals.append(f"Sentiment: {sentiment}")
    if importance_score > 0:
        signals.append(f"Importance: {importance_score:.1f}/100")
    if signals:
        summary = f"{summary} ({' | '.join(signals)})" if summary else " | ".join(signals)
    return _truncate_text(summary, max_length)


def _normalize_token(token: str) -> str:
    return token.strip("'-").lower()


def _is_keyword_candidate(word: str) -> bool:
    return (
        (len(word) >= 3 or word in _SHORT_KEYWORDS)
        and word not in _STOPWORDS
        and not word.isnumeric()
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max(0, max_length - 3)].rstrip() + "..."
