from __future__ import annotations

from ai_pulse_tracker.enrichment import compute_importance_score, extract_keywords


def test_extract_keywords_ranks_repeated_terms():
    keywords = extract_keywords(
        "AI agents transform enterprise automation",
        "Enterprise teams adopt AI agents for automation and analytics.",
    )

    assert keywords[:2] == ["ai", "agents"]
    assert "automation" in keywords
    assert "enterprise" in keywords


def test_extract_keywords_ignores_common_words():
    keywords = extract_keywords(
        "AI pour les équipes data",
        "Les équipes utilisent AI pour analyser des données.",
    )

    assert "pour" not in keywords
    assert "les" not in keywords
    assert "équipes" in keywords


def test_compute_importance_score_uses_confidence_and_completeness():
    score = compute_importance_score(
        title="AI agents transform enterprise automation",
        description="A detailed article about AI agent adoption.",
        url="https://example.com/article",
        source="Example",
        sentiment_confidence=0.9,
        keywords=["ai", "agents", "automation", "enterprise"],
    )

    assert score == 87.0
