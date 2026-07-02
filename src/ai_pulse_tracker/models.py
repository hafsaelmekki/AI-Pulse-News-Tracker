from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import hashlib
from typing import Any, Dict
from urllib.parse import urlparse


@dataclass(slots=True)
class Article:
    source: str
    title: str
    description: str | None
    url: str
    published_at: datetime

    def document_id(self) -> str:
        if self.url:
            return _encode_url_as_id(self.url)
        return _hash_fallback_id(self.title, self.published_at)

    def partition_key(self) -> str:
        return _compute_partition_key(self.url, self.source or self.title)


@dataclass(slots=True)
class AnalyzedArticle(Article):
    sentiment: str
    confidence_pos: float
    confidence_neu: float
    confidence_neg: float
    keywords: list[str] = field(default_factory=list)
    importance_score: float = 0.0
    summary: str = ""

    def cosmos_id(self) -> str:
        return self.document_id()

    def to_cosmos_document(self) -> Dict[str, Any]:
        return {
            "id": self.cosmos_id(),
            "source": self.partition_key(),
            "source_name": self.source,
            "title": self.title,
            "description": self.description or "",
            "sentiment": self.sentiment,
            "confidence": {
                "pos": self.confidence_pos,
                "neu": self.confidence_neu,
                "neg": self.confidence_neg,
            },
            "keywords": self.keywords,
            "importance_score": self.importance_score,
            "summary": self.summary,
            "url": self.url,
            "date": self.published_at.isoformat(),
        }


@dataclass(slots=True)
class UpsertResult:
    ids: list[str]
    created: int = 0
    updated: int = 0

    def __len__(self) -> int:
        return len(self.ids)


def _encode_url_as_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")
    return f"url__{encoded}"


def _hash_fallback_id(title: str, published_at: datetime) -> str:
    base = (title or "untitled").strip()
    timestamp = _ensure_timezone(published_at).isoformat()
    payload = f"{base}|{timestamp}".encode("utf-8")
    return "hash__" + hashlib.sha1(payload).hexdigest()


def _compute_partition_key(url: str | None, fallback: str) -> str:
    if url:
        host = urlparse(url).netloc.lower()
        if host:
            return host
    sanitized = (fallback or "unknown").strip().lower()
    return sanitized or "unknown"


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
