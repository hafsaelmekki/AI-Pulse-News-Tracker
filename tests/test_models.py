from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib

from ai_pulse_tracker.models import AnalyzedArticle


def _make_article() -> AnalyzedArticle:
    return AnalyzedArticle(
        source="Test",
        title="Sample",
        description="Desc",
        url="https://example.com",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        sentiment="positive",
        confidence_pos=0.8,
        confidence_neu=0.15,
        confidence_neg=0.05,
        keywords=["sample", "desc"],
        importance_score=82.5,
    )


def test_cosmos_id_uses_url():
    article = _make_article()
    encoded = base64.urlsafe_b64encode(article.url.encode("utf-8")).decode("ascii").rstrip("=")
    assert article.cosmos_id() == f"url__{encoded}"


def test_to_cosmos_document_includes_id():
    article = _make_article()
    doc = article.to_cosmos_document()
    assert doc["id"] == article.cosmos_id()
    assert doc["source"] == "example.com"
    assert doc["source_name"] == "Test"
    assert doc["description"] == "Desc"
    assert doc["keywords"] == ["sample", "desc"]
    assert doc["importance_score"] == 82.5


def test_partition_key_falls_back_to_source_when_url_missing():
    article = _make_article()
    article.url = ""
    assert article.partition_key() == "test"


def test_hash_fallback_used_when_url_missing():
    article = _make_article()
    article.url = ""
    expected = hashlib.sha1(
        f"{article.title}|{article.published_at.isoformat()}".encode("utf-8")
    ).hexdigest()
    assert article.cosmos_id() == f"hash__{expected}"
