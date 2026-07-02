from __future__ import annotations

from ai_pulse_tracker.agent import answer_question


def _results() -> list[dict]:
    return [
        {
            "source": "Tech News",
            "title": "OpenAI launches agent tools",
            "summary": "Enterprise teams use AI agents to automate workflows.",
            "matched_terms": "ai, agents, automation",
            "sentiment": "positive",
            "keywords": ["ai", "agents", "automation"],
            "url": "https://example.com/agents",
        },
        {
            "source": "AI Weekly",
            "title": "RAG systems move into production",
            "summary": "Teams combine retrieval and generation for internal knowledge.",
            "matched_terms": "rag",
            "sentiment": "neutral",
            "keywords": ["rag", "retrieval"],
            "url": "https://example.com/rag",
        },
    ]


def test_answer_question_uses_retrieved_articles_and_citations():
    answer = answer_question("What are the AI trends?", _results())

    assert "Answer based on 2 retrieved article" in answer
    assert "Main signals" not in answer
    assert "[1]" in answer
    assert "[2]" in answer
    assert "Tech News" in answer
    assert "https://example.com/agents" in answer


def test_answer_question_handles_empty_results():
    answer = answer_question("What are the AI trends?", [])

    assert "could not find stored articles" in answer
