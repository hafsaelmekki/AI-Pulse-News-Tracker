from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import os
from typing import Any

import requests

from .trends import normalize_keywords


AI_PULSE_SYSTEM_PROMPT = """
You are AI Pulse Assistant, a professional AI market intelligence analyst.

Your job is to answer questions using only the retrieved AI Pulse articles.
Use a concise executive style: direct answer first, then key insights, then evidence.

Pipeline:
Cosmos DB articles -> retrieval -> structured article context -> LLM -> analytical answer with sources.

Rules:
- Stay strictly within AI Pulse dashboard data and retrieved AI Pulse articles.
- If the user asks about an unrelated topic, politely say you can only analyze AI Pulse data.
- Do not invent facts that are not present in the retrieved articles.
- Mention uncertainty when the evidence is limited.
- Use citations like [1], [2], [3] when referring to articles.
- Prioritize business relevance: trends, companies, risks, opportunities, and signals.
- Keep the answer clear, polished, and recruiter/demo friendly.
- Always include a sources section with source names and URLs when URLs are available.
- If the user asks a conversational or language-switching question, answer naturally
  without using retrieved articles.
""".strip()

LLM_TIMEOUT_SECONDS = 20

CONVERSATION_STARTERS = {
    "hello",
    "hey",
    "hi",
    "bonjour",
    "bonsoir",
    "salut",
    "coucou",
}

META_PROMPTS = {
    "can we speak",
    "can we talk",
    "speak english",
    "speak in english",
    "talk english",
    "talk in english",
    "parler anglais",
    "en anglais",
    "tu peux parler",
    "what can you do",
    "who are you",
    "help me",
}

DATA_QUERY_TERMS = {
    "ai",
    "ia",
    "rag",
    "llm",
    "genai",
    "agent",
    "agents",
    "article",
    "articles",
    "news",
    "trend",
    "trends",
    "tendance",
    "tendances",
    "source",
    "sources",
    "company",
    "companies",
    "entreprise",
    "entreprises",
    "openai",
    "anthropic",
    "claude",
    "sentiment",
    "signal",
    "signals",
    "signaux",
    "dashboard",
    "importance",
    "important",
    "regulation",
    "automation",
    "automatisation",
}


def is_conversation_prompt(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return True

    first_word = normalized.replace(",", " ").replace("!", " ").split(maxsplit=1)[0]
    if first_word in CONVERSATION_STARTERS:
        return True
    if any(phrase in normalized for phrase in META_PROMPTS):
        return True
    return not _is_data_query(normalized)


def answer_conversation(question: str) -> str:
    normalized = question.strip().lower()
    first_word = normalized.replace(",", " ").replace("!", " ").split(maxsplit=1)[0]
    if _asks_for_english(normalized):
        return (
            "Yes Hafsa, we can speak in English. "
            "I stay focused on AI Pulse only: dashboard trends, RAG, AI agents, "
            "companies, sources, sentiment, important articles, and weak signals."
        )
    if not normalized or first_word in CONVERSATION_STARTERS:
        return (
            "Hey Hafsa. Je suis ton assistant AI Pulse. "
            "Je peux analyser les tendances IA, les companies, les sources, "
            "le sentiment, l'importance des articles et les signaux faibles du dashboard. "
            "Choisis une question ci-dessous ou pose ta propre question."
        )
    return (
        "Je peux uniquement rester dans le périmètre AI Pulse. "
        "Pose-moi une question sur les tendances, sentiment, sources, companies, "
        "articles importants ou signaux faibles du dashboard."
    )


def answer_question(question: str, retrieved_articles: list[Mapping[str, Any]]) -> str:
    """Cosmos DB articles -> retrieval -> article context -> LLM -> sourced answer."""
    question = question.strip()
    if is_conversation_prompt(question):
        return answer_conversation(question)

    if not retrieved_articles:
        return (
            f"I could not find stored articles that answer: {question}. "
            "Try a broader question or ingest more recent articles."
        )

    article_context = _article_context(retrieved_articles)
    messages = build_llm_messages(question, article_context)
    try:
        llm_answer = _try_generate_with_llm(messages)
    except RuntimeError as exc:
        return (
            "LLM is required for assistant answers, but it is not available. "
            f"{exc}"
        )

    return _ensure_source_appendix(llm_answer, retrieved_articles)


def build_llm_messages(
    question: str,
    article_context: list[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized_context = (
        list(article_context)
        if article_context and "rank" in article_context[0]
        else _article_context(article_context)
    )
    context_lines = [
        "Retrieved AI Pulse articles:",
        "",
    ]
    for article in normalized_context:
        context_lines.extend(
            [
                f"[{article['rank']}] {article['title']}",
                f"Source: {article['source']}",
                f"Sentiment: {article['sentiment']}",
                f"Importance score: {article['importance_score']}",
                f"Summary: {article['summary']}",
                f"URL: {article['url']}",
                "",
            ]
        )

    user_prompt = "\n".join(
        [
            "\n".join(context_lines).strip(),
            "",
            f"User question: {question}",
            "",
            (
                "Generate a professional analytical answer grounded only in the "
                "articles above. Include source citations and source URLs."
            ),
        ]
    )
    return [
        {"role": "system", "content": AI_PULSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _try_generate_with_llm(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("AI_PULSE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set AI_PULSE_LLM_API_KEY or OPENAI_API_KEY to enable the assistant."
        )

    endpoint = os.getenv(
        "AI_PULSE_LLM_ENDPOINT",
        "https://api.openai.com/v1/chat/completions",
    )
    model = os.getenv("AI_PULSE_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("LLM response format was invalid.") from exc

    answer = str(answer).strip()
    if not answer:
        raise RuntimeError("LLM returned an empty answer.")
    return answer


def _article_context(retrieved_articles: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for index, article in enumerate(retrieved_articles, start=1):
        context.append(
            {
                "rank": index,
                "title": str(article.get("title", "")).strip() or "Untitled article",
                "summary": str(article.get("summary", "")).strip()
                or str(article.get("description", "")).strip()
                or "No summary available.",
                "sentiment": str(article.get("sentiment", "")).strip() or "unknown",
                "importance_score": _format_importance(article.get("importance_score")),
                "source": str(article.get("source", "")).strip() or "Unknown source",
                "url": str(article.get("url", "")).strip() or "No URL available.",
            }
        )
    return context


def _professional_fallback_answer(
    question: str,
    retrieved_articles: list[Mapping[str, Any]],
) -> str:
    citations = _citation_labels(retrieved_articles)
    keywords = _top_keywords(retrieved_articles)
    sentiment_counts = Counter(
        str(article.get("sentiment", "")).strip().lower()
        for article in retrieved_articles
        if str(article.get("sentiment", "")).strip()
    )
    dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else "unknown"
    focus = ", ".join(keywords[:5]) if keywords else "the retrieved articles"

    lines = [
        "### Executive answer",
        (
            f"The available AI Pulse evidence points to {focus} as the main signal for "
            f"your question. The dominant sentiment across the retrieved articles is "
            f"{dominant_sentiment}."
        ),
        "",
        "### Key insights",
    ]
    lines.extend(_insight_lines(retrieved_articles, citations))
    lines.extend(
        [
            "",
            "### Evidence",
        ]
    )
    lines.extend(_source_lines(retrieved_articles, citations))
    lines.extend(
        [
            "",
            "### Suggested next step",
            f"Use this answer as a first-pass market signal, then review the cited sources before making a decision about: {question}",
        ]
    )
    return "\n".join(lines)


def _ensure_source_appendix(
    answer: str,
    retrieved_articles: list[Mapping[str, Any]],
) -> str:
    source_urls = [
        str(article.get("url", "")).strip()
        for article in retrieved_articles
        if str(article.get("url", "")).strip()
    ]
    if source_urls and any(url in answer for url in source_urls):
        return answer

    citations = _citation_labels(retrieved_articles)
    source_lines = _source_lines(retrieved_articles, citations)
    if not source_lines:
        return answer

    return "\n".join(
        [
            answer.rstrip(),
            "",
            "### Sources used",
            *source_lines,
        ]
    )


def _overview_sentence(
    question: str,
    keywords: list[str],
    dominant_sentiment: str,
    citations: list[str],
) -> str:
    topic_text = ", ".join(keywords[:5]) if keywords else "the retrieved articles"
    cited = " ".join(citations[:3])
    return (
        f"For '{question}', the strongest signals are {topic_text}. "
        f"The dominant sentiment in the retrieved set is {dominant_sentiment}. {cited}"
    ).strip()


def _signal_lines(
    retrieved_articles: list[Mapping[str, Any]],
    citations: list[str],
) -> list[str]:
    lines: list[str] = []
    for article, citation in zip(retrieved_articles[:3], citations):
        title = str(article.get("title", "")).strip() or "Untitled article"
        summary = str(article.get("summary", "")).strip()
        matched_terms = str(article.get("matched_terms", "")).strip()
        signal = summary or title
        suffix = f" Matched terms: {matched_terms}." if matched_terms else ""
        lines.append(f"- {signal}{suffix} {citation}")
    return lines


def _insight_lines(
    retrieved_articles: list[Mapping[str, Any]],
    citations: list[str],
) -> list[str]:
    lines: list[str] = []
    for article, citation in zip(retrieved_articles[:3], citations):
        title = str(article.get("title", "")).strip() or "Untitled article"
        summary = str(article.get("summary", "")).strip()
        sentiment = str(article.get("sentiment", "")).strip() or "unknown"
        importance = _format_importance(article.get("importance_score"))
        source = str(article.get("source", "")).strip() or "Unknown source"
        signal = summary or title
        lines.append(
            f"- {signal} {citation} Source: {source}. "
            f"Sentiment: {sentiment}. Importance: {importance}."
        )
    return lines


def _source_lines(
    retrieved_articles: list[Mapping[str, Any]],
    citations: list[str],
) -> list[str]:
    lines: list[str] = []
    for article, citation in zip(retrieved_articles, citations):
        source = str(article.get("source", "")).strip() or "Unknown source"
        title = str(article.get("title", "")).strip() or "Untitled article"
        sentiment = str(article.get("sentiment", "")).strip() or "unknown"
        importance = _format_importance(article.get("importance_score"))
        url = str(article.get("url", "")).strip()
        if url:
            lines.append(
                f"- {citation} {source}: {title} "
                f"| Sentiment: {sentiment} | Importance: {importance} | {url}"
            )
        else:
            lines.append(
                f"- {citation} {source}: {title} "
                f"| Sentiment: {sentiment} | Importance: {importance}"
            )
    return lines


def _top_keywords(retrieved_articles: list[Mapping[str, Any]], limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for article in retrieved_articles:
        counter.update(normalize_keywords(article.get("keywords")))
    return [keyword for keyword, _ in counter.most_common(limit)]


def _citation_labels(retrieved_articles: list[Mapping[str, Any]]) -> list[str]:
    return [f"[{index}]" for index in range(1, len(retrieved_articles) + 1)]


def _asks_for_english(normalized_question: str) -> bool:
    return any(
        phrase in normalized_question
        for phrase in {
            "english",
            "anglais",
            "en anglais",
        }
    )


def _is_data_query(normalized_question: str) -> bool:
    words = set(normalized_question.replace("?", " ").replace(",", " ").split())
    return any(term in words or term in normalized_question for term in DATA_QUERY_TERMS)


def _format_importance(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "unknown"
