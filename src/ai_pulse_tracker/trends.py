from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_keywords = value.split(",")
    elif isinstance(value, Iterable):
        raw_keywords = value
    else:
        return []

    keywords: list[str] = []
    for keyword in raw_keywords:
        normalized = str(keyword).strip().lower()
        if normalized:
            keywords.append(normalized)
    return keywords


def count_keywords(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, int | str]]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(normalize_keywords(row.get("keywords")))

    return [
        {"keyword": keyword, "count": count}
        for keyword, count in counter.most_common(max(1, limit))
    ]


def format_keywords(value: Any) -> str:
    return ", ".join(normalize_keywords(value))
