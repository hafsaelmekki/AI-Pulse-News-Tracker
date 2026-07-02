from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .trends import format_keywords, normalize_keywords


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+'-]*")
_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "avec",
    "dans",
    "des",
    "for",
    "from",
    "how",
    "les",
    "plus",
    "pour",
    "que",
    "qui",
    "the",
    "this",
    "une",
    "what",
    "which",
    "with",
}
_SHORT_TERMS = {"ai", "ia", "ml"}


def search_articles(
    rows: Iterable[Mapping[str, Any]],
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    query_counter = Counter(query_terms)
    results: list[dict[str, Any]] = []
    for row in rows:
        result = _score_row(row, query, query_counter)
        if result["score"] > 0:
            results.append(result)

    results.sort(
        key=lambda result: (
            -float(result["score"]),
            -float(result["importance_score"]),
            str(result["date"]),
        )
    )
    return [
        {"rank": index + 1, **result}
        for index, result in enumerate(results[: max(1, limit)])
    ]


def build_retrieval_answer(query: str, results: list[Mapping[str, Any]]) -> str:
    if not results:
        return (
            f"No stored article matched: {query}. Try a broader topic such as "
            "'AI agents', 'OpenAI', 'RAG' or 'automation'."
        )

    top_titles = [str(result.get("title", "")).strip() for result in results[:3]]
    top_titles = [title for title in top_titles if title]
    sources = sorted(
        {
            str(result.get("source", "")).strip()
            for result in results
            if str(result.get("source", "")).strip()
        }
    )
    keywords = Counter()
    for result in results:
        keywords.update(normalize_keywords(result.get("keywords")))
    top_keywords = [keyword for keyword, _ in keywords.most_common(5)]

    parts = [
        f"Retrieved {len(results)} relevant article(s) for: {query}.",
    ]
    if top_keywords:
        parts.append("Main signals: " + ", ".join(top_keywords) + ".")
    if sources:
        parts.append("Sources: " + ", ".join(sources[:5]) + ".")
    if top_titles:
        parts.append("Most relevant: " + " | ".join(top_titles) + ".")
    return " ".join(parts)


def _score_row(
    row: Mapping[str, Any],
    query: str,
    query_counter: Counter[str],
) -> dict[str, Any]:
    title = str(row.get("title") or "")
    summary = str(row.get("summary") or "")
    description = str(row.get("description") or "")
    source = str(row.get("source") or row.get("source_name") or "")
    keywords = normalize_keywords(row.get("keywords"))
    importance_score = _safe_float(row.get("importance_score"))

    title_terms = set(_tokenize(title))
    keyword_terms = set(_tokenize(" ".join(keywords)))
    body_terms = Counter(_tokenize(" ".join([title, summary, description, source])))
    matched_terms = sorted(set(query_counter).intersection(body_terms))
    if not matched_terms:
        return _result(row, 0.0, [], keywords, source, importance_score)

    overlap_score = sum(min(query_counter[term], body_terms[term]) for term in matched_terms) * 10
    title_score = len(set(query_counter).intersection(title_terms)) * 5
    keyword_score = len(set(query_counter).intersection(keyword_terms)) * 7
    phrase_score = 15 if query.strip().lower() in _search_text(row).lower() else 0
    score = overlap_score + title_score + keyword_score + phrase_score + importance_score / 20
    return _result(row, round(score, 2), matched_terms, keywords, source, importance_score)


def _result(
    row: Mapping[str, Any],
    score: float,
    matched_terms: list[str],
    keywords: list[str],
    source: str,
    importance_score: float,
) -> dict[str, Any]:
    return {
        "score": score,
        "matched_terms": ", ".join(matched_terms),
        "date": row.get("date", ""),
        "source": source,
        "title": row.get("title", ""),
        "summary": row.get("summary", ""),
        "sentiment": row.get("sentiment", ""),
        "importance_score": importance_score,
        "keywords": keywords,
        "keyword_text": format_keywords(keywords),
        "url": row.get("url", ""),
    }


def _search_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("summary") or ""),
            str(row.get("description") or ""),
            str(row.get("source") or row.get("source_name") or ""),
            " ".join(normalize_keywords(row.get("keywords"))),
        ]
    )


def _tokenize(text: str) -> list[str]:
    terms = [term.lower().strip("'-") for term in _TOKEN_RE.findall(text)]
    return [
        term
        for term in terms
        if (len(term) >= 3 or term in _SHORT_TERMS)
        and term not in _STOPWORDS
        and not term.isnumeric()
    ]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
