from __future__ import annotations

from ai_pulse_tracker.agent import (
    AI_PULSE_SYSTEM_PROMPT,
    answer_question,
    build_llm_messages,
    is_conversation_prompt,
)


def _results() -> list[dict]:
    return [
        {
            "source": "Tech News",
            "title": "OpenAI launches agent tools",
            "summary": "Enterprise teams use AI agents to automate workflows.",
            "matched_terms": "ai, agents, automation",
            "sentiment": "positive",
            "importance_score": 86.0,
            "keywords": ["ai", "agents", "automation"],
            "url": "https://example.com/agents",
        },
        {
            "source": "AI Weekly",
            "title": "RAG systems move into production",
            "summary": "Teams combine retrieval and generation for internal knowledge.",
            "matched_terms": "rag",
            "sentiment": "neutral",
            "importance_score": 72.5,
            "keywords": ["rag", "retrieval"],
            "url": "https://example.com/rag",
        },
    ]


def test_answer_question_uses_professional_grounded_format(monkeypatch):
    monkeypatch.delenv("AI_PULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    answer = answer_question("What are the AI trends?", _results())

    assert "### Executive answer" in answer
    assert "### Key insights" in answer
    assert "### Evidence" in answer
    assert "Answer based on" not in answer
    assert "Main signals" not in answer
    assert "[1]" in answer
    assert "[2]" in answer
    assert "Tech News" in answer
    assert "Importance: 86.0" in answer
    assert "https://example.com/agents" in answer


def test_answer_question_handles_empty_results():
    answer = answer_question("What are the AI trends?", [])

    assert "could not find stored articles" in answer


def test_answer_question_handles_conversation_prompts():
    answer = answer_question("hey Hafsa", [])

    assert is_conversation_prompt("hey Hafsa")
    assert "Hey Hafsa" in answer
    assert "could not find stored articles" not in answer


def test_answer_question_does_not_search_language_switch_prompt():
    answer = answer_question("can we speak in english", _results())

    assert is_conversation_prompt("can we speak in english")
    assert "Yes Hafsa" in answer
    assert "strongest signals" not in answer


def test_build_llm_messages_contains_article_context():
    messages = build_llm_messages("What are the AI trends?", _results())

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == AI_PULSE_SYSTEM_PROMPT
    assert "Cosmos DB articles -> retrieval -> structured article context -> LLM" in messages[0]["content"]
    assert "OpenAI launches agent tools" in messages[1]["content"]
    assert "Enterprise teams use AI agents to automate workflows." in messages[1]["content"]
    assert "Sentiment: positive" in messages[1]["content"]
    assert "Importance score: 86.0" in messages[1]["content"]
    assert "Source: Tech News" in messages[1]["content"]
    assert "URL: https://example.com/agents" in messages[1]["content"]


def test_answer_question_appends_sources_to_llm_answer(monkeypatch):
    monkeypatch.setattr(
        "ai_pulse_tracker.agent._try_generate_with_llm",
        lambda messages: "### Executive answer\nAI agents are the main signal [1].",
    )

    answer = answer_question("What are the AI agent trends?", _results())

    assert "### Executive answer" in answer
    assert "### Sources used" in answer
    assert "Tech News" in answer
    assert "https://example.com/agents" in answer
