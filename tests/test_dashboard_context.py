from __future__ import annotations

import pandas as pd

from ai_pulse_tracker.dashboard import (
    build_compact_dashboard_context,
    build_dashboard_context,
)


def test_build_dashboard_context_adds_missing_sentiment_score():
    df = pd.DataFrame(
        [
            {
                "date": "2026-07-04T10:31:00+00:00",
                "title": "OpenAI launches AI agent tools",
                "source": "AI News",
                "sentiment": "positive",
                "confidence": {"pos": 0.8, "neu": 0.15, "neg": 0.05},
                "importance_score": 86.0,
                "summary": "Enterprise teams use AI agents.",
                "description": "Enterprise teams use AI agents.",
                "url": "https://example.com/ai",
                "topic": "AI Agents",
                "topics": ["AI Agents"],
                "keywords": ["ai", "agents"],
            }
        ]
    )

    context = build_dashboard_context(df)

    assert context["top_sources"][0]["Source"] == "AI News"
    assert context["top_sources"][0]["Avg sentiment"] == "+0.75"


def test_build_compact_dashboard_context_limits_articles_and_text():
    rows = []
    for day in range(20):
        rows.append(
            {
                "date": f"2026-07-{day + 1:02d}T10:31:00+00:00",
                "title": "OpenAI launches AI agent tools " + ("x" * 400),
                "source": "AI News",
                "sentiment": "negative" if day % 2 else "positive",
                "confidence": {"pos": 0.8, "neu": 0.15, "neg": 0.05},
                "importance_score": 95.0 - day,
                "summary": "Enterprise teams use AI agents. " + ("y" * 500),
                "description": "Long description " + ("z" * 500),
                "url": f"https://example.com/ai-{day}",
                "topic": "AI Agents",
                "topics": ["AI Agents"],
                "keywords": ["ai", "agents"],
            }
        )
    context = build_compact_dashboard_context(pd.DataFrame(rows))

    assert len(context["top_important_articles"]) == 5
    assert len(context["negative_high_importance_articles"]) == 5
    assert "url" not in context["top_important_articles"][0]
    assert len(context["top_important_articles"][0]["title"]) <= 220
    assert len(context["top_important_articles"][0]["summary"]) <= 220
    assert len({record["date"][:10] for record in context["sentiment_by_day"]}) == 14
