from __future__ import annotations

from ai_pulse_tracker.relevance import ai_relevance_reason, is_ai_related


def test_is_ai_related_matches_explicit_ai_terms():
    article = {
        "title": "OpenAI launches new ChatGPT agent tools",
        "description": "Enterprise teams use generative AI for automation.",
        "content": "",
    }

    assert is_ai_related(article)
    assert ai_relevance_reason(article) == "matched keyword: generative ai"


def test_is_ai_related_rejects_unrelated_local_news():
    article = {
        "title": "Hotel deal announced for summer vacations",
        "description": "Local tourism teams expect more visitors.",
        "content": "The offer includes rooms and restaurants.",
    }

    assert not is_ai_related(article)
    assert ai_relevance_reason(article) == ""


def test_is_ai_related_matches_short_ai_as_word_only():
    assert is_ai_related({"title": "New AI model announced"})
    assert not is_ai_related({"title": "Daily train delays continue"})
