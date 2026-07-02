from __future__ import annotations

from ai_pulse_tracker.retrieval import build_retrieval_answer, search_articles


def _rows() -> list[dict]:
    return [
        {
            "date": "2026-07-03T08:00:00+00:00",
            "source": "Tech News",
            "title": "OpenAI launches new AI agent tools",
            "summary": "Enterprise teams use AI agents to automate workflows.",
            "description": "The announcement focuses on automation and agent orchestration.",
            "sentiment": "positive",
            "importance_score": 90,
            "keywords": ["ai", "agents", "automation"],
            "url": "https://example.com/agents",
        },
        {
            "date": "2026-07-03T09:00:00+00:00",
            "source": "Cloud Daily",
            "title": "Cloud infrastructure pricing update",
            "summary": "Cloud providers change pricing for storage services.",
            "description": "Infrastructure teams compare storage and compute costs.",
            "sentiment": "neutral",
            "importance_score": 40,
            "keywords": ["cloud", "pricing"],
            "url": "https://example.com/cloud",
        },
    ]


def test_search_articles_returns_ranked_matches():
    results = search_articles(_rows(), "AI agent automation")

    assert results[0]["rank"] == 1
    assert results[0]["title"] == "OpenAI launches new AI agent tools"
    assert "automation" in results[0]["matched_terms"]
    assert results[0]["score"] > 0


def test_search_articles_returns_empty_for_unmatched_query():
    assert search_articles(_rows(), "quantum biology") == []


def test_build_retrieval_answer_summarizes_results():
    results = search_articles(_rows(), "AI agent automation")
    answer = build_retrieval_answer("AI agent automation", results)

    assert "Retrieved 1 relevant article" in answer
    assert "Main signals" in answer
    assert "Tech News" in answer
