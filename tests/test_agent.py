from __future__ import annotations

from ai_pulse_tracker.agent import (
    AI_PULSE_SYSTEM_PROMPT,
    MAX_LLM_PAYLOAD_KB,
    _payload_size_kb,
    _protected_dashboard_payload,
    answer_conversation,
    answer_dashboard_question,
    answer_important_articles_by_company,
    answer_question,
    build_dashboard_llm_messages,
    build_hybrid_assistant_context,
    build_llm_messages,
    build_retrieved_evidence,
    detect_assistant_language,
    detect_dashboard_intent,
    detect_language_change_request,
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


def _dashboard_context() -> dict:
    return {
        "total_articles": 10,
        "date_range": {"start": "2026-07-01", "end": "2026-07-05"},
        "sentiment_distribution": [
            {"sentiment": "neutral", "articles": 6},
            {"sentiment": "positive", "articles": 3},
            {"sentiment": "negative", "articles": 1},
        ],
        "average_importance_score": 81.2,
        "dominant_sentiment": "neutral",
        "top_companies": [
            {"Company": "OpenAI", "Articles": 8, "Avg importance": 86.4, "Dominant sentiment": "Neutral"},
            {"Company": "Google", "Articles": 6, "Avg importance": 78.1, "Dominant sentiment": "Positive"},
        ],
        "top_sources": [
            {"Source": "AI News", "articles": 7, "Avg importance": 82.0},
            {"Source": "Tech News", "articles": 3, "Avg importance": 75.0},
        ],
        "top_topics": [
            {"topic": "AI Agents", "articles": 5, "avg_importance": 84.0},
            {"topic": "RAG", "articles": 3, "avg_importance": 79.0},
        ],
        "top_important_articles": [
            {
                "title": "OpenAI launches important AI agent update",
                "source": "AI News",
                "summary": "OpenAI agent tools for enterprise workflows.",
                "sentiment": "neutral",
                "importance_score": 92.4,
                "keywords": ["openai", "agents"],
                "companies": ["OpenAI"],
            },
            {
                "title": "Google Gemini expands enterprise AI deployment",
                "source": "Tech News",
                "summary": "Google Gemini adoption grows.",
                "sentiment": "positive",
                "importance_score": 88.1,
                "keywords": ["google", "gemini"],
                "companies": ["Google"],
            },
        ],
        "negative_high_importance_articles": [],
        "weak_signals": [
            {"Topic": "AI Privacy", "Articles": 2, "Avg importance": "88.0"},
        ],
    }


def test_answer_question_requires_llm_when_missing(monkeypatch):
    monkeypatch.delenv("AI_PULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    answer = answer_question("What are the AI trends?", _results())

    assert "LLM is required for assistant answers" in answer
    assert "AI_PULSE_LLM_API_KEY" in answer
    assert "OPENAI_API_KEY" in answer
    assert "### Executive answer" not in answer


def test_answer_question_handles_empty_results():
    answer = answer_question("What are the AI trends?", [])

    assert "could not find stored articles" in answer


def test_answer_question_handles_conversation_prompts():
    answer = answer_question("hey Hafsa", [])

    assert is_conversation_prompt("hey Hafsa")
    assert "Hi Hafsa" in answer
    assert "could not find stored articles" not in answer


def test_answer_question_does_not_search_language_switch_prompt():
    answer = answer_question("can we speak in english", _results())

    assert is_conversation_prompt("can we speak in english")
    assert "Yes Hafsa" in answer
    assert "strongest signals" not in answer


def test_assistant_language_detection_and_switch_requests():
    assert detect_assistant_language("bonjour") == "fr"
    assert detect_assistant_language("bonjou") == "fr"
    assert detect_assistant_language("bjr") == "fr"
    assert detect_assistant_language("Bonjour, ça va ?") == "fr"
    assert detect_assistant_language("je veux voir les tendances") == "fr"
    assert detect_assistant_language("show companies") == "en"
    assert detect_assistant_language("show me companies") == "en"
    assert detect_language_change_request("réponds en anglais") == "en"
    assert detect_language_change_request("switch to French") == "fr"
    assert detect_language_change_request("show companies") is None


def test_answer_conversation_uses_latest_message_language():
    english_answer = answer_conversation("bonjour", language="en")
    french_answer = answer_conversation("hello", language="fr")
    typo_answer = answer_conversation("Bonjou")
    casual_answer = answer_conversation("Bonjour, ça va ?")

    assert "Hi Hafsa" in english_answer
    assert "Bonjour Hafsa" in french_answer
    assert "Bonjour Hafsa" in typo_answer
    assert "Bonjour Hafsa" in casual_answer


def test_build_llm_messages_contains_article_context():
    messages = build_llm_messages("What are the AI trends?", _results())

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == AI_PULSE_SYSTEM_PROMPT
    assert "AI dashboard analyst" in messages[0]["content"]
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


def test_detect_dashboard_intent_maps_dashboard_questions():
    assert detect_dashboard_intent("What changed in sentiment over time?") == "sentiment"
    assert detect_dashboard_intent("companies mentioned today") == "companies"
    assert detect_dashboard_intent("most important articles") == "importance"
    assert detect_dashboard_intent("weak signals") == "weak_signals"
    assert detect_dashboard_intent("last week") == "follow_up"


def test_build_dashboard_llm_messages_contains_dashboard_context():
    context = {
        "total_articles": 2,
        "top_companies": [{"Company": "OpenAI", "Articles": 2}],
    }
    messages = build_dashboard_llm_messages(
        "Which companies are most mentioned?",
        context,
        _results(),
        intent="companies",
    )

    assert messages[0]["content"] == AI_PULSE_SYSTEM_PROMPT
    assert "hybrid_assistant_context" in messages[1]["content"]
    assert "dashboard_context" in messages[1]["content"]
    assert "retrieved_evidence" in messages[1]["content"]
    assert '"total_articles": 2' in messages[1]["content"]
    assert '"citation_id": "[1]"' in messages[1]["content"]


def test_build_dashboard_llm_messages_uses_locked_language():
    messages = build_dashboard_llm_messages(
        "What changed in sentiment?",
        {"total_articles": 2},
        [],
        intent="sentiment",
        language="fr",
    )

    assert "Answer language: French" in messages[1]["content"]
    assert "Write the entire answer in French" in messages[1]["content"]
    assert "detailed RAG-style analytical answer" in messages[1]["content"]
    assert "current dashboard filters" in messages[1]["content"]


def test_build_retrieved_evidence_is_compact_and_cited():
    evidence = build_retrieved_evidence(_results(), max_articles=1)

    assert len(evidence) == 1
    assert evidence[0]["citation_id"] == "[1]"
    assert evidence[0]["title"] == "OpenAI launches agent tools"
    assert evidence[0]["published_at"] == ""
    assert evidence[0]["importance_score"] == "86.0"
    assert evidence[0]["companies"] == ["OpenAI"]
    assert len(evidence[0]["short_summary"]) <= 400
    assert "content" not in evidence[0]


def test_build_hybrid_assistant_context_combines_dashboard_and_evidence():
    context = build_hybrid_assistant_context(
        "What are the trends?",
        _dashboard_context(),
        _results(),
    )

    assert context["question"] == "What are the trends?"
    assert context["dashboard_context"]["total_articles"] == 10
    assert context["retrieved_evidence"][0]["citation_id"] == "[1]"
    assert "Use dashboard_context" in context["instructions"]


def test_answer_dashboard_question_uses_llm_when_available(monkeypatch):
    monkeypatch.delenv("AI_PULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "ai_pulse_tracker.agent._try_generate_with_llm",
        lambda messages: "### Executive summary\nOpenAI is the strongest company signal [1].",
    )

    answer = answer_dashboard_question(
        "Which companies are most mentioned?",
        _dashboard_context(),
        _results(),
        intent="companies",
    )

    assert "OpenAI" in answer
    assert "### Executive summary" in answer
    assert "### Sources used" in answer
    assert "https://example.com/agents" in answer
    assert "LLM is required" not in answer


def test_simple_dashboard_intents_use_llm_when_available(monkeypatch):
    calls = []

    def fake_llm(messages):
        calls.append(messages)
        return "### Executive summary\nThis is a RAG-style dashboard answer [1]."

    monkeypatch.setattr(
        "ai_pulse_tracker.agent._try_generate_with_llm",
        fake_llm,
    )

    cases = [
        ("Which sources are most active?", "sources"),
        ("What is the sentiment distribution?", "sentiment"),
        ("Which articles are most important?", "importance"),
    ]
    for question, intent in cases:
        answer = answer_dashboard_question(
            question,
            _dashboard_context(),
            _results(),
            intent=intent,
        )
        assert "RAG-style dashboard answer" in answer
        assert "LLM is required" not in answer
    assert len(calls) == 3


def test_deterministic_dashboard_answer_uses_locked_language(monkeypatch):
    monkeypatch.delenv("AI_PULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    answer = answer_dashboard_question(
        "Which sources are most active?",
        _dashboard_context(),
        [],
        intent="sources",
        language="fr",
    )

    assert "synthèse RAG générative est temporairement indisponible" in answer
    assert "### Résumé exécutif" in answer
    assert "AI News" in answer


def test_answer_important_articles_by_company_matches_top_companies():
    answer = answer_important_articles_by_company(_dashboard_context())

    assert "OpenAI launches important AI agent update" in answer
    assert "OpenAI, AI News" in answer
    assert "importance 92.4/100" in answer
    assert "Google Gemini expands enterprise AI deployment" in answer


def test_answer_dashboard_question_falls_back_on_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "ai_pulse_tracker.agent._try_generate_with_llm",
        lambda messages: (_ for _ in ()).throw(
            RuntimeError("LLM provider is rate-limited. Using deterministic AI Pulse fallback.")
        ),
    )

    answer = answer_dashboard_question(
        "What are the strongest AI Pulse trends?",
        _dashboard_context(),
        [],
        intent="trends",
    )

    assert "### Executive summary" in answer
    assert "AI Agents" in answer
    assert "LLM is required" not in answer


def test_protected_dashboard_payload_reduces_oversized_context():
    huge_text = "OpenAI AI agents " + ("x" * 6000)
    context = {
        "total_articles": 20,
        "date_range": {"start": "2026-07-01", "end": "2026-07-20"},
        "sentiment_distribution": [{"sentiment": "positive", "articles": 12}],
        "average_importance_score": 82.5,
        "dominant_sentiment": "positive",
        "top_important_articles": [
            {
                "title": huge_text,
                "summary": huge_text,
                "description": huge_text,
                "url": "https://example.com/large",
                "importance_score": 95.0,
            }
            for _ in range(5)
        ],
        "negative_high_importance_articles": [
            {
                "title": huge_text,
                "summary": huge_text,
                "description": huge_text,
                "url": "https://example.com/large-negative",
                "importance_score": 91.0,
            }
            for _ in range(5)
        ],
        "top_companies": [{"Company": "OpenAI", "Articles": 8}],
        "top_sources": [{"Source": "AI News", "articles": 8}],
        "top_topics": [{"topic": "AI Agents", "articles": 8}],
    }

    payload = _protected_dashboard_payload(
        question="Which articles are most important?",
        dashboard_context=context,
        retrieved_articles=_results(),
        intent="importance",
    )

    assert _payload_size_kb(payload) <= MAX_LLM_PAYLOAD_KB
    articles = payload["dashboard_context"].get("top_important_articles", [])
    assert len(articles) <= 3
    assert all("url" not in article for article in articles)
    assert all("description" not in article for article in articles)
