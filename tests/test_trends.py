from __future__ import annotations

from ai_pulse_tracker.trends import count_keywords, format_keywords, normalize_keywords


def test_normalize_keywords_accepts_lists_and_strings():
    assert normalize_keywords(["AI", " Agents ", ""]) == ["ai", "agents"]
    assert normalize_keywords("AI, Agents, RAG") == ["ai", "agents", "rag"]


def test_count_keywords_ranks_by_frequency():
    rows = [
        {"keywords": ["ai", "agents", "rag"]},
        {"keywords": ["ai", "automation"]},
        {"keywords": "agents, ai"},
    ]

    assert count_keywords(rows, limit=2) == [
        {"keyword": "ai", "count": 3},
        {"keyword": "agents", "count": 2},
    ]


def test_format_keywords_returns_display_text():
    assert format_keywords(["AI", "RAG"]) == "ai, rag"
    assert format_keywords(None) == ""
