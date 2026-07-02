from __future__ import annotations

from ai_pulse_tracker.embeddings import article_embedding_text, cosine_similarity, embed_text


def test_embed_text_is_deterministic_and_normalized():
    first = embed_text("AI agents automate workflows")
    second = embed_text("AI agents automate workflows")

    assert first == second
    assert round(cosine_similarity(first, first), 6) == 1.0


def test_cosine_similarity_rewards_related_text():
    query = embed_text("AI agent automation")
    related = embed_text("AI agents automate enterprise workflows")
    unrelated = embed_text("cloud storage pricing update")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_article_embedding_text_combines_retrieval_fields():
    text = article_embedding_text(
        title="AI agents",
        summary="Workflow automation",
        description="Enterprise adoption",
        source="Tech News",
        keywords=["ai", "agents"],
    )

    assert "AI agents" in text
    assert "Workflow automation" in text
    assert "Enterprise adoption" in text
    assert "Tech News" in text
    assert "ai agents" in text
