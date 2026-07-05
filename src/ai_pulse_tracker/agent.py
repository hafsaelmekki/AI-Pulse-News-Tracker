from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import os
import time
from typing import Any

import requests

from .trends import normalize_keywords


AI_PULSE_SYSTEM_PROMPT = """
You are AI Pulse, a RAG-powered AI dashboard analyst.

Your role is to analyze the AI Pulse dashboard and explain the news signals behind the numbers.

You have two sources of truth:
1. dashboard_context:
   - aggregated metrics
   - sentiment distribution
   - companies
   - sources
   - importance scores
   - trends over time
   - weak signals

2. retrieved_evidence:
   - articles retrieved from the database
   - each article has a citation id such as [1], [2], [3]

Rules:
- Use dashboard_context for all numerical statements.
- Use retrieved_evidence to justify and illustrate the analysis.
- Do not invent numbers.
- Do not invent sources.
- Do not invent articles.
- Do not cite an article unless it is present in retrieved_evidence.
- Do not simply list articles.
- Always synthesize.
- Answer in the same language as the user.
- If the question is in French, answer fully in French.
- If the question is in English, answer fully in English.
- Stay inside the AI / Generative AI scope.
- If some retrieved articles are weak or only indirectly relevant, say so clearly.
- Never confuse sentiment label with importance_score.
- Sentiment is a label: positive, neutral, negative or mixed.
- Importance score is a priority score from 0 to 100.
- Be detailed, but structured.
- Use citations like [1], [2], [3] after claims supported by retrieved articles.

Answer format for analytical questions:
1. Résumé exécutif / Executive summary
2. Analyse détaillée / Detailed analysis
3. Ce que cela signifie / What it means
4. Articles de référence / Evidence articles
5. Limites éventuelles / Limitations, only if useful
""".strip()

LLM_TIMEOUT_SECONDS = 20
LLM_MAX_RETRIES = 2
LLM_RETRY_BASE_SECONDS = 0.75
MAX_LLM_PAYLOAD_KB = 28.0
EVIDENCE_TEXT_LIMIT = 220

CONVERSATION_STARTERS = {
    "bjr",
    "bonjou",
    "hello",
    "hey",
    "hi",
    "bonjour",
    "bonsoir",
    "salut",
    "coucou",
}

CASUAL_CONVERSATION_PHRASES = {
    "ca va",
    "ça va",
    "comment ca va",
    "comment ça va",
    "how are you",
    "how is it going",
    "merci",
    "thanks",
    "thank you",
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
    "today",
    "yesterday",
    "week",
    "days",
    "aujourd'hui",
    "aujourdhui",
    "hier",
    "semaine",
}

INTENT_KEYWORDS = {
    "sentiment": ("sentiment", "positive", "negative", "neutral", "positif", "négatif", "negatif"),
    "companies": ("company", "companies", "entreprise", "entreprises", "openai", "google", "microsoft", "anthropic", "mistral", "nvidia"),
    "sources": ("source", "sources", "media", "publisher", "publication"),
    "importance": ("importance", "important", "priority", "priorité", "priorite", "score", "article", "articles"),
    "weak_signals": ("weak signal", "weak signals", "signal faible", "signaux faibles", "risk", "risque", "risks"),
    "trends": ("trend", "trends", "tendance", "tendances", "topic", "topics", "rag", "agent", "agents", "llm", "genai"),
}

TEMPORAL_FOLLOW_UP_TERMS = (
    "today",
    "yesterday",
    "last week",
    "this week",
    "last 7 days",
    "ces derniers jours",
    "aujourd'hui",
    "aujourdhui",
    "hier",
    "la semaine dernière",
    "la semaine derniere",
)


def is_conversation_prompt(question: str) -> bool:
    normalized = _normalize_question(question)
    if not normalized:
        return True

    if _is_greeting_or_casual(normalized):
        return True
    if any(phrase in normalized for phrase in META_PROMPTS):
        return True
    return not _is_data_query(normalized)


def detect_dashboard_intent(question: str) -> str:
    normalized = _normalize_question(question)
    if not normalized:
        return "general_summary"
    if _is_temporal_follow_up(normalized):
        return "follow_up"
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return intent
    return "general_summary"


def detect_assistant_language(question: str) -> str:
    return _answer_language(question)


def detect_language_change_request(question: str) -> str | None:
    normalized = _normalize_question(question)
    english_phrases = {
        "english",
        "in english",
        "speak english",
        "speak in english",
        "switch to english",
        "change to english",
        "answer in english",
        "reply in english",
        "anglais",
        "en anglais",
        "parle anglais",
        "parler anglais",
        "repond en anglais",
        "répond en anglais",
        "reponds en anglais",
        "réponds en anglais",
    }
    french_phrases = {
        "french",
        "in french",
        "speak french",
        "speak in french",
        "switch to french",
        "change to french",
        "answer in french",
        "reply in french",
        "francais",
        "français",
        "en francais",
        "en français",
        "parle francais",
        "parle français",
        "parler francais",
        "parler français",
        "repond en francais",
        "repond en français",
        "répond en francais",
        "répond en français",
        "reponds en francais",
        "reponds en français",
        "réponds en francais",
        "réponds en français",
    }
    if any(phrase in normalized for phrase in english_phrases):
        return "en"
    if any(phrase in normalized for phrase in french_phrases):
        return "fr"
    return None


def answer_conversation(question: str, language: str | None = None) -> str:
    normalized = _normalize_question(question)
    target_language = language or detect_language_change_request(question) or _answer_language(question)
    requested_language = detect_language_change_request(question)
    if requested_language == "en":
        return (
            "Yes Hafsa, we can speak in English. "
            "I stay focused on AI Pulse only: dashboard trends, RAG, AI agents, "
            "companies, sources, sentiment, important articles, and weak signals."
        )
    if requested_language == "fr":
        return (
            "Oui Hafsa, on peut parler en français. "
            "Je reste concentré uniquement sur AI Pulse : tendances du dashboard, RAG, "
            "agents IA, companies, sources, sentiment, articles importants et signaux faibles."
        )
    if not normalized or _is_greeting_or_casual(normalized):
        return _localized(
            target_language,
            (
                "Hi Hafsa — I’m here. Ask me about AI Pulse trends, companies, "
                "sentiment, sources, important articles, or weak signals."
            ),
            (
                "Bonjour Hafsa — ça va. Pose-moi une question sur AI Pulse : "
                "tendances, companies, sentiment, sources, articles importants ou signaux faibles."
            ),
        )
    return _localized(
        target_language,
        (
            "I can only stay within the AI Pulse scope. "
            "Ask me about dashboard trends, sentiment, sources, companies, "
            "important articles, or weak signals."
        ),
        (
            "Je peux uniquement rester dans le périmètre AI Pulse. "
            "Pose-moi une question sur les tendances, sentiment, sources, companies, "
            "articles importants ou signaux faibles du dashboard."
        ),
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


def answer_dashboard_question(
    question: str,
    dashboard_context: Mapping[str, Any],
    retrieved_articles: list[Mapping[str, Any]],
    *,
    intent: str,
    language: str | None = None,
) -> str:
    messages = build_dashboard_llm_messages(
        question,
        dashboard_context,
        retrieved_articles,
        intent=intent,
        language=language,
    )
    try:
        llm_answer = _try_generate_with_llm(messages)
    except RuntimeError as exc:
        return _fallback_dashboard_answer(
            question,
            dashboard_context,
            intent=intent,
            error=str(exc),
            language=language,
        )
    return _ensure_source_appendix(llm_answer, retrieved_articles, language=language or "en")


def answer_important_articles_by_company(
    dashboard_context: Mapping[str, Any],
    language: str = "en",
) -> str:
    top_companies = _top_company_names(dashboard_context)
    if not top_companies:
        return _localized(
            language,
            "I do not have enough AI Pulse company data to answer this.",
            "Je n’ai pas assez de données company dans AI Pulse pour répondre.",
        )

    articles = list(dashboard_context.get("top_important_articles", []))
    articles.extend(dashboard_context.get("negative_high_importance_articles", []))
    matches: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for article in articles:
        if not isinstance(article, Mapping):
            continue
        matched_company = _matched_company(article, top_companies)
        title = str(article.get("title", "")).strip()
        if not matched_company or title in seen_titles:
            continue
        seen_titles.add(title)
        matches.append({**dict(article), "matched_company": matched_company})

    matches.sort(key=lambda article: _safe_float(article.get("importance_score")), reverse=True)
    if not matches:
        return _localized(
            language,
            "No top important AI Pulse articles explicitly mention the main dashboard companies.",
            "Aucun article important AI Pulse ne mentionne explicitement les principales companies du dashboard.",
        )

    top_match = matches[0]
    summary = _localized(
        language,
        (
            "The strongest important-article signal is linked to "
            f"{top_match.get('matched_company', 'the main companies')}."
        ),
        (
            "Le signal le plus important côté articles est lié à "
            f"{top_match.get('matched_company', 'les principales companies')}."
        ),
    )
    insights: list[str] = []
    key_numbers = _base_key_numbers(dashboard_context, language)
    for article in matches[:5]:
        title = str(article.get("title", "Untitled article")).strip() or "Untitled article"
        insights.append(
            f"{title} — {article.get('matched_company', 'unknown')}, "
            f"{article.get('source', 'Unknown source')}, "
            f"sentiment {article.get('sentiment', 'unknown')}, "
            f"importance {_format_importance(article.get('importance_score'))}/100."
        )
    key_numbers.append(
        _localized(
            language,
            f"Matched important articles: {len(matches[:5])}",
            f"Articles importants détectés : {len(matches[:5])}",
        )
    )
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=key_numbers,
        sources=_source_names_from_articles(matches),
    )


def build_retrieved_evidence(
    retrieved_articles: list[Mapping[str, Any]],
    max_articles: int = 6,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, article in enumerate(retrieved_articles[: max(1, max_articles)], start=1):
        topics = _article_topics_or_keywords(article)
        evidence.append(
            {
                "citation_id": f"[{index}]",
                "title": _truncate_text(article.get("title") or "Untitled article", 180),
                "source": _truncate_text(article.get("source") or "Unknown source", 120),
                "published_at": _truncate_text(
                    article.get("published_at") or article.get("date") or "",
                    80,
                ),
                "url": _truncate_text(article.get("url") or "", 300),
                "sentiment": _truncate_text(article.get("sentiment") or "unknown", 40),
                "importance_score": _format_importance(article.get("importance_score")),
                "companies": _article_companies(article),
                "topics": topics,
                "keywords": topics,
                "short_summary": _truncate_text(
                    article.get("summary") or article.get("description") or "",
                    400,
                ),
            }
        )
    return evidence


def build_hybrid_assistant_context(
    question: str,
    dashboard_context: Mapping[str, Any],
    retrieved_articles: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "question": question,
        "dashboard_context": dict(dashboard_context),
        "retrieved_evidence": build_retrieved_evidence(retrieved_articles),
        "instructions": (
            "Use dashboard_context for quantitative claims and retrieved_evidence "
            "for source-backed explanations."
        ),
    }


def build_dashboard_llm_messages(
    question: str,
    dashboard_context: Mapping[str, Any],
    retrieved_articles: list[Mapping[str, Any]],
    *,
    intent: str,
    language: str | None = None,
) -> list[dict[str, str]]:
    payload = _protected_dashboard_payload(
        question=question,
        dashboard_context=dashboard_context,
        retrieved_articles=retrieved_articles,
        intent=intent,
        language=language,
    )
    target_language = language or _answer_language(question)
    language_name = "French" if target_language == "fr" else "English"
    context_json = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    user_prompt = "\n".join(
        [
            f"Dashboard intent: {intent}",
            f"Answer language: {language_name}",
            f"Payload size: {payload['payload_size_kb']:.1f} KB",
            "",
            "hybrid_assistant_context:",
            context_json,
            "",
            f"User question: {question}",
            "",
            (
                "Produce a detailed RAG-style analytical answer. Use dashboard_context "
                "first for metrics and retrieved_evidence for citations. "
                f"Write the entire answer in {language_name}. Mention that the answer "
                "is based on the current dashboard filters or selected date range. "
                "For trends, identify 3 to 5 trends when possible. For companies, "
                "compare visibility, average importance and dominant sentiment. "
                "For sentiment, explain distribution and drivers. For important articles, "
                "rank by importance_score. For weak signals, explain low-volume but "
                "high-importance signals. End with evidence articles and citation ids."
            ),
        ]
    )
    return [
        {"role": "system", "content": AI_PULSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _deterministic_dashboard_answer(
    question: str,
    dashboard_context: Mapping[str, Any],
    *,
    intent: str,
    language: str | None = None,
) -> str | None:
    language = language or _answer_language(question)
    normalized = _normalize_question(question)
    if _asks_important_articles_by_company(normalized):
        return answer_important_articles_by_company(dashboard_context, language=language)
    if intent == "companies":
        return _answer_companies(dashboard_context, language)
    if intent == "sources":
        return _answer_sources(dashboard_context, language)
    if intent == "sentiment":
        return _answer_sentiment(dashboard_context, language)
    if intent == "importance":
        return _answer_importance(dashboard_context, language)
    return None


def _fallback_dashboard_answer(
    question: str,
    dashboard_context: Mapping[str, Any],
    *,
    intent: str,
    error: str,
    language: str | None = None,
) -> str:
    language = language or _answer_language(question)
    deterministic = _deterministic_dashboard_answer(
        question,
        dashboard_context,
        intent=intent,
        language=language,
    )
    if deterministic:
        return _prepend_unavailable_llm_notice(deterministic, language)
    if intent == "weak_signals":
        return _prepend_unavailable_llm_notice(
            _answer_weak_signals(dashboard_context, language, error),
            language,
        )
    if intent == "trends":
        return _prepend_unavailable_llm_notice(
            _answer_trends(dashboard_context, language, error),
            language,
        )
    return _prepend_unavailable_llm_notice(
        _answer_general_summary(dashboard_context, language, error),
        language,
    )


def _prepend_unavailable_llm_notice(answer: str, language: str) -> str:
    notice = _localized(
        language,
        "Generative RAG synthesis is temporarily unavailable, so this is a dashboard-based fallback.",
        "La synthèse RAG générative est temporairement indisponible ; voici une réponse basée sur le dashboard.",
    )
    return f"{notice}\n\n{answer}"


def _answer_companies(dashboard_context: Mapping[str, Any], language: str) -> str:
    companies = _as_records(dashboard_context.get("top_companies"))[:10]
    if not companies:
        return _localized(
            language,
            "I do not have enough AI Pulse company data to answer this.",
            "Je n’ai pas assez de données company dans AI Pulse pour répondre.",
        )
    strongest = companies[0]
    strongest_name = strongest.get("Company") or strongest.get("company") or "the top company"
    summary = _localized(
        language,
        f"{strongest_name} is the strongest company signal in the current dashboard context.",
        f"{strongest_name} est le signal company le plus fort dans le contexte dashboard actuel.",
    )
    insights: list[str] = []
    for company in companies[:3]:
        name = company.get("Company") or company.get("company") or "Unknown"
        articles = company.get("Articles") or company.get("articles") or 0
        avg_importance = company.get("Avg importance") or company.get("avg_importance") or "unknown"
        sentiment = company.get("Dominant sentiment") or company.get("dominant_sentiment") or "unknown"
        insights.append(
            f"{name}: {articles} articles, average importance {avg_importance}, dominant sentiment {sentiment}."
        )
    key_numbers = _base_key_numbers(dashboard_context, language)
    key_numbers.append(
        _localized(
            language,
            f"Companies tracked in top list: {len(companies)}",
            f"Companies dans le top : {len(companies)}",
        )
    )
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=key_numbers,
    )


def _answer_sources(dashboard_context: Mapping[str, Any], language: str) -> str:
    sources = _as_records(dashboard_context.get("top_sources"))[:10]
    if not sources:
        return _localized(
            language,
            "I do not have enough AI Pulse source data to answer this.",
            "Je n’ai pas assez de données sources dans AI Pulse pour répondre.",
        )
    top_source = sources[0].get("Source") or sources[0].get("source") or "the top source"
    summary = _localized(
        language,
        f"{top_source} is the most active source in the current dashboard context.",
        f"{top_source} est la source la plus active dans le contexte dashboard actuel.",
    )
    insights: list[str] = []
    for source in sources[:3]:
        name = source.get("Source") or source.get("source") or "Unknown"
        articles = source.get("articles") or source.get("Articles") or 0
        avg_importance = source.get("Avg importance") or source.get("avg_importance") or "unknown"
        insights.append(f"{name}: {articles} articles, average importance {avg_importance}.")
    key_numbers = _base_key_numbers(dashboard_context, language)
    key_numbers.append(
        _localized(language, f"Top sources shown: {len(sources)}", f"Sources dans le top : {len(sources)}")
    )
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=key_numbers,
    )


def _answer_sentiment(dashboard_context: Mapping[str, Any], language: str) -> str:
    distribution = _as_records(dashboard_context.get("sentiment_distribution"))
    sentiment_by_day = _as_records(dashboard_context.get("sentiment_by_day"))[-9:]
    dominant = dashboard_context.get("dominant_sentiment", "unknown")
    if not distribution:
        return _localized(
            language,
            "I do not have enough AI Pulse sentiment data to answer this.",
            "Je n’ai pas assez de données de sentiment AI Pulse pour répondre.",
        )
    summary = _localized(
        language,
        f"The dominant AI Pulse sentiment is {dominant} in the current dashboard context.",
        f"Le sentiment dominant dans AI Pulse est {dominant} dans le contexte dashboard actuel.",
    )
    insights = [
        f"{item.get('sentiment', 'unknown')}: {item.get('articles', 0)} articles"
        for item in distribution[:3]
    ]
    if sentiment_by_day:
        insights.append(
            _localized(
                language,
                "Recent daily sentiment percentages are available for the latest dashboard days.",
                "Les pourcentages quotidiens récents sont disponibles pour les derniers jours du dashboard.",
            )
        )
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=_base_key_numbers(dashboard_context, language),
    )


def _answer_importance(dashboard_context: Mapping[str, Any], language: str) -> str:
    articles = _as_records(dashboard_context.get("top_important_articles"))[:5]
    average_importance = dashboard_context.get("average_importance_score", 0.0)
    if not articles:
        return _localized(
            language,
            "I do not have enough AI Pulse importance data to answer this.",
            "Je n’ai pas assez de données d’importance AI Pulse pour répondre.",
        )
    summary = _localized(
        language,
        f"The average AI Pulse importance score is {average_importance}.",
        f"Le score d’importance moyen AI Pulse est {average_importance}.",
    )
    insights = []
    for article in articles[:3]:
        insights.append(
            f"{article.get('title', 'Untitled article')} — "
            f"{article.get('source', 'Unknown source')}, "
            f"sentiment {article.get('sentiment', 'unknown')}, "
            f"importance {_format_importance(article.get('importance_score'))}/100."
        )
    key_numbers = _base_key_numbers(dashboard_context, language)
    key_numbers.append(
        _localized(language, f"Top important articles shown: {len(articles)}", f"Articles importants affichés : {len(articles)}")
    )
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=key_numbers,
        sources=_source_names_from_articles(articles),
    )


def _answer_weak_signals(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    signals = _as_records(dashboard_context.get("weak_signals"))[:6]
    if not signals:
        summary = _fallback_notice(
            language,
            "No weak signals are currently visible in the compact AI Pulse dashboard context.",
            "Aucun signal faible n’est visible dans le contexte compact AI Pulse.",
            error,
        )
        return _format_dashboard_answer(
            language=language,
            dashboard_context=dashboard_context,
            summary=summary,
            insights=[_localized(language, "No low-volume high-importance topic stands out.", "Aucun topic peu visible et très important ne ressort.")],
            key_numbers=_base_key_numbers(dashboard_context, language),
        )
    summary = _fallback_notice(
        language,
        "These weak signals come directly from dashboard aggregations.",
        "Ces signaux faibles viennent directement des agrégations dashboard.",
        error,
    )
    insights = []
    for signal in signals[:3]:
        topic = signal.get("Topic") or signal.get("topic") or "Unknown"
        articles = signal.get("Articles") or signal.get("articles") or 0
        avg_importance = signal.get("Avg importance") or signal.get("avg_importance") or "unknown"
        insights.append(f"{topic}: {articles} articles, average importance {avg_importance}.")
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=_base_key_numbers(dashboard_context, language),
    )


def _answer_trends(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    topics = _as_records(dashboard_context.get("top_topics"))[:5]
    if not topics:
        return _answer_general_summary(dashboard_context, language, error)
    top_topic = topics[0].get("topic") or topics[0].get("Topic") or "the leading topic"
    summary = _fallback_notice(
        language,
        f"{top_topic} is the strongest visible trend in the current dashboard context.",
        f"{top_topic} est la tendance la plus visible dans le contexte dashboard actuel.",
        error,
    )
    insights = []
    for topic in topics[:3]:
        name = topic.get("topic") or topic.get("Topic") or "Unknown"
        articles = topic.get("articles") or topic.get("Articles") or 0
        avg_importance = topic.get("avg_importance") or topic.get("Avg importance") or "unknown"
        insights.append(f"{name}: {articles} articles, average importance {avg_importance}.")
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=_base_key_numbers(dashboard_context, language),
    )


def _answer_general_summary(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    total_articles = dashboard_context.get("total_articles", 0)
    date_range = dashboard_context.get("date_range", {})
    dominant = dashboard_context.get("dominant_sentiment", "unknown")
    average_importance = dashboard_context.get("average_importance_score", 0.0)
    summary = _fallback_notice(
        language,
        f"AI Pulse currently contains {total_articles} articles in the active dashboard context.",
        f"AI Pulse contient actuellement {total_articles} articles dans le contexte dashboard actif.",
        error,
    )
    insights = [
        _localized(
            language,
            f"Dominant sentiment is {dominant}.",
            f"Le sentiment dominant est {dominant}.",
        ),
        _localized(
            language,
            f"Average importance is {average_importance}.",
            f"L’importance moyenne est {average_importance}.",
        ),
        _localized(
            language,
            f"Date range is {date_range.get('start')} → {date_range.get('end')}.",
            f"La période est {date_range.get('start')} → {date_range.get('end')}.",
        ),
    ]
    return _format_dashboard_answer(
        language=language,
        dashboard_context=dashboard_context,
        summary=summary,
        insights=insights,
        key_numbers=_base_key_numbers(dashboard_context, language),
    )


def _fallback_notice(language: str, english: str, french: str, error: str) -> str:
    notice = _localized(language, english, french)
    if "rate-limited" in error.lower() or "429" in error:
        return notice
    return notice


def _format_dashboard_answer(
    *,
    language: str,
    dashboard_context: Mapping[str, Any],
    summary: str,
    insights: list[str],
    key_numbers: list[str],
    sources: list[str] | None = None,
) -> str:
    headings = {
        "en": ("Executive summary", "Top insights", "Key numbers", "Sources used"),
        "fr": ("Résumé exécutif", "Insights clés", "Chiffres clés", "Sources utilisées"),
    }
    executive_heading, insights_heading, numbers_heading, sources_heading = headings.get(
        language,
        headings["en"],
    )
    scope_line = _dashboard_scope_line(dashboard_context, language)
    source_names = sources or _source_names_from_context(dashboard_context)
    if not source_names:
        source_names = [
            _localized(
                language,
                "Current dashboard context",
                "Contexte dashboard actuel",
            )
        ]

    lines = [
        f"### {executive_heading}",
        scope_line,
        summary,
        "",
        f"### {insights_heading}",
    ]
    lines.extend(f"- {insight}" for insight in insights[:3])
    lines.extend(["", f"### {numbers_heading}"])
    lines.extend(f"- {number}" for number in key_numbers[:5])
    lines.extend(["", f"### {sources_heading}"])
    lines.extend(f"- {source}" for source in source_names[:5])
    return "\n".join(lines).strip()


def _dashboard_scope_line(dashboard_context: Mapping[str, Any], language: str) -> str:
    temporal_filter = dashboard_context.get("temporal_filter")
    filter_context = dashboard_context.get("filter_context")
    if isinstance(filter_context, Mapping):
        date_label = str(filter_context.get("date_label") or "").strip()
        date_range = filter_context.get("date_range")
        sentiments = _filter_value_summary(filter_context.get("sentiments"), language)
        sources = _filter_value_summary(filter_context.get("sources"), language)
        min_importance = filter_context.get("min_importance")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_label = f"{date_label} ({date_range[0]} → {date_range[1]})".strip()
        temporal_text = f", demande temporelle={temporal_filter}" if temporal_filter else ""
        if language == "fr":
            return (
                "Réponse basée sur les filtres dashboard actuels : "
                f"dates={date_label or 'contexte actuel'}, sentiments={sentiments}, "
                f"sources={sources}, importance min={min_importance}{temporal_text}."
            )
        temporal_text = f", time request={temporal_filter}" if temporal_filter else ""
        return (
            "Based on the current dashboard filters: "
            f"dates={date_label or 'current context'}, sentiments={sentiments}, "
            f"sources={sources}, min importance={min_importance}{temporal_text}."
        )

    date_range = dashboard_context.get("date_range")
    if isinstance(date_range, Mapping) and (date_range.get("start") or date_range.get("end")):
        if language == "fr":
            return (
                "Réponse basée sur le contexte dashboard actuel "
                f"({date_range.get('start')} → {date_range.get('end')})"
                f"{f', demande temporelle={temporal_filter}' if temporal_filter else ''}."
            )
        return (
            "Based on the current dashboard context "
            f"({date_range.get('start')} → {date_range.get('end')})"
            f"{f', time request={temporal_filter}' if temporal_filter else ''}."
        )
    return _localized(
        language,
        "Based on the current dashboard context.",
        "Réponse basée sur le contexte dashboard actuel.",
    )


def _filter_value_summary(value: Any, language: str) -> str:
    if isinstance(value, list):
        if not value:
            return _localized(language, "all", "tous")
        if len(value) <= 3:
            return ", ".join(str(item) for item in value)
        return _localized(language, f"{len(value)} selected", f"{len(value)} sélectionnés")
    if value:
        return str(value)
    return _localized(language, "all", "tous")


def _base_key_numbers(dashboard_context: Mapping[str, Any], language: str) -> list[str]:
    date_range = dashboard_context.get("date_range", {})
    date_text = (
        f"{date_range.get('start')} → {date_range.get('end')}"
        if isinstance(date_range, Mapping)
        else "unknown"
    )
    if language == "fr":
        return [
            f"Articles analysés : {dashboard_context.get('total_articles', 0)}",
            f"Période : {date_text}",
            f"Sentiment dominant : {dashboard_context.get('dominant_sentiment', 'unknown')}",
            f"Importance moyenne : {dashboard_context.get('average_importance_score', 0.0)}",
        ]
    return [
        f"Analyzed articles: {dashboard_context.get('total_articles', 0)}",
        f"Date range: {date_text}",
        f"Dominant sentiment: {dashboard_context.get('dominant_sentiment', 'unknown')}",
        f"Average importance: {dashboard_context.get('average_importance_score', 0.0)}",
    ]


def _source_names_from_context(dashboard_context: Mapping[str, Any]) -> list[str]:
    sources: list[str] = []
    for source in _as_records(dashboard_context.get("top_sources"))[:5]:
        name = source.get("Source") or source.get("source")
        if name:
            sources.append(str(name))
    return sources


def _source_names_from_articles(articles: list[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for article in articles:
        source = str(article.get("source", "")).strip()
        if source and source not in names:
            names.append(source)
    return names


def _article_topics_or_keywords(article: Mapping[str, Any]) -> list[str]:
    topics = _coerce_text_list(article.get("topics"))
    topic = str(article.get("topic") or "").strip()
    if topic:
        topics.append(topic)
    keywords = normalize_keywords(article.get("keywords"))
    merged: list[str] = []
    for value in [*topics, *keywords]:
        clean_value = str(value).strip()
        if clean_value and clean_value not in merged:
            merged.append(_truncate_text(clean_value, 80))
    return merged[:8]


def _article_companies(article: Mapping[str, Any]) -> list[str]:
    explicit_companies = _coerce_text_list(article.get("companies"))
    extracted_entities = _coerce_text_list(article.get("extracted_entities"))
    text = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("summary") or ""),
            str(article.get("description") or ""),
            " ".join(normalize_keywords(article.get("keywords"))),
            " ".join(explicit_companies),
            " ".join(extracted_entities),
        ]
    ).lower()
    known_companies = {
        "OpenAI": ("openai", "chatgpt", "gpt-"),
        "Google": ("google", "gemini", "deepmind"),
        "Microsoft": ("microsoft", "azure", "copilot"),
        "Meta": ("meta", "llama"),
        "Anthropic": ("anthropic", "claude"),
        "Mistral AI": ("mistral",),
        "Nvidia": ("nvidia",),
        "Apple": ("apple",),
    }
    companies: list[str] = []
    for company in explicit_companies:
        clean_company = _truncate_text(company, 80)
        if clean_company and clean_company not in companies:
            companies.append(clean_company)
    for company, aliases in known_companies.items():
        if any(alias in text for alias in aliases) and company not in companies:
            companies.append(company)
    return companies[:8]


def _coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return []


def _asks_important_articles_by_company(normalized_question: str) -> bool:
    return (
        ("mention" in normalized_question or "cite" in normalized_question)
        and ("company" in normalized_question or "companies" in normalized_question or "entreprise" in normalized_question)
        and ("important" in normalized_question or "importance" in normalized_question or "article" in normalized_question)
    )


def _top_company_names(dashboard_context: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for company in _as_records(dashboard_context.get("top_companies")):
        name = company.get("Company") or company.get("company")
        if name:
            names.append(str(name))
    return names


def _matched_company(article: Mapping[str, Any], companies: list[str]) -> str | None:
    search_parts = [
        article.get("title", ""),
        article.get("summary", ""),
        " ".join(str(keyword) for keyword in article.get("keywords", []) or []),
        " ".join(str(company) for company in article.get("companies", []) or []),
        " ".join(str(entity) for entity in article.get("extracted_entities", []) or []),
    ]
    haystack = " ".join(str(part or "") for part in search_parts).lower()
    for company in companies:
        if company.lower() in haystack:
            return company
    return None


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _localized(language: str, english: str, french: str) -> str:
    return french if language == "fr" else english


def _answer_language(question: str) -> str:
    normalized = _normalize_question(question)
    french_words = {
        "je",
        "tu",
        "moi",
        "nous",
        "veux",
        "peux",
        "peut",
        "pourquoi",
        "comment",
        "dans",
        "avec",
        "sans",
        "les",
        "des",
        "une",
        "sur",
        "ce",
        "ça",
        "ca",
        "ajoute",
        "enleve",
        "enlève",
        "bjr",
        "bonjour",
        "bonjou",
        "bonsoir",
        "salut",
        "coucou",
    }
    french_phrases = {
        "quels",
        "quelle",
        "quelles",
        "entreprise",
        "entreprises",
        "tendance",
        "tendances",
        "signaux",
        "faibles",
        "aujourd'hui",
        "hier",
        "semaine",
    }
    words = set(normalized.replace("?", " ").replace(",", " ").replace(".", " ").split())
    if words & french_words or any(marker in normalized for marker in french_phrases):
        return "fr"
    return "en"


def _protected_dashboard_payload(
    *,
    question: str,
    dashboard_context: Mapping[str, Any],
    retrieved_articles: list[Mapping[str, Any]],
    intent: str,
    language: str | None = None,
) -> dict[str, Any]:
    payload = {
        "intent": intent,
        "answer_language": "French" if (language or _answer_language(question)) == "fr" else "English",
        **build_hybrid_assistant_context(
            question,
            dashboard_context,
            retrieved_articles,
        ),
    }
    if _payload_size_kb(payload) <= MAX_LLM_PAYLOAD_KB:
        payload["payload_size_kb"] = _payload_size_kb(payload)
        return payload

    reduced_payload = _reduce_dashboard_payload(payload)
    if _payload_size_kb(reduced_payload) <= MAX_LLM_PAYLOAD_KB:
        reduced_payload["payload_size_kb"] = _payload_size_kb(reduced_payload)
        return reduced_payload

    minimal_payload = _minimal_dashboard_payload(reduced_payload)
    minimal_payload["payload_size_kb"] = _payload_size_kb(minimal_payload)
    return minimal_payload


def _payload_size_kb(payload: Mapping[str, Any]) -> float:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")) / 1024


def _reduce_dashboard_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    reduced_context = _without_keys(
        dict(payload.get("dashboard_context", {})),
        {"url", "description", "content"},
    )
    for key in ("top_important_articles", "negative_high_importance_articles"):
        if isinstance(reduced_context.get(key), list):
            reduced_context[key] = reduced_context[key][:3]
    reduced_evidence = [
        _without_keys(dict(article), {"description", "content"})
        for article in list(payload.get("retrieved_evidence", []))[:4]
    ]
    return {
        "intent": payload.get("intent"),
        "answer_language": payload.get("answer_language"),
        "question": payload.get("question"),
        "dashboard_context": reduced_context,
        "retrieved_evidence": reduced_evidence,
        "instructions": payload.get("instructions"),
    }


def _minimal_dashboard_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = dict(payload.get("dashboard_context", {}))
    minimal_context = {
        key: context.get(key)
        for key in (
            "total_articles",
            "date_range",
            "sentiment_distribution",
            "average_importance_score",
            "dominant_sentiment",
            "top_companies",
            "top_sources",
            "top_topics",
            "sentiment_by_day",
            "weak_signals",
            "temporal_filter",
        )
        if key in context
    }
    for key in ("top_companies", "top_sources", "top_topics", "sentiment_by_day"):
        if isinstance(minimal_context.get(key), list):
            minimal_context[key] = minimal_context[key][:5]
    return {
        "intent": payload.get("intent"),
        "answer_language": payload.get("answer_language"),
        "question": payload.get("question"),
        "dashboard_context": minimal_context,
        "retrieved_evidence": [
            _without_keys(dict(article), {"url", "short_summary", "description", "content"})
            for article in list(payload.get("retrieved_evidence", []))[:3]
        ],
        "instructions": payload.get("instructions"),
    }


def _without_keys(value: Any, keys_to_remove: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_keys(item, keys_to_remove)
            for key, item in value.items()
            if key not in keys_to_remove
        }
    if isinstance(value, list):
        return [_without_keys(item, keys_to_remove) for item in value]
    if isinstance(value, str):
        return _truncate_text(value, EVIDENCE_TEXT_LIMIT)
    return value


def _compact_evidence_article(article: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": article.get("rank"),
        "title": _truncate_text(article.get("title") or "Untitled article"),
        "source": _truncate_text(article.get("source") or "Unknown source", 120),
        "summary": _truncate_text(
            article.get("summary") or article.get("description") or "No summary available."
        ),
        "sentiment": _truncate_text(article.get("sentiment") or "unknown", 40),
        "importance_score": _format_importance(article.get("importance_score")),
        "matched_terms": _truncate_text(article.get("matched_terms") or "", 120),
        "url": _truncate_text(article.get("url") or "", 300),
    }


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

    last_error: requests.RequestException | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
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
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                if response.status_code == 429 and attempt < LLM_MAX_RETRIES:
                    time.sleep(LLM_RETRY_BASE_SECONDS * (2**attempt))
                    last_error = exc
                    continue
                if response.status_code == 429:
                    raise RuntimeError(
                        "LLM provider is rate-limited. Using deterministic AI Pulse fallback."
                    ) from exc
                raise
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"]
            break
        except requests.RequestException as exc:
            last_error = exc
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("LLM response format was invalid.") from exc
    else:
        raise RuntimeError(f"LLM request failed: {last_error}")

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
                "title": _truncate_text(
                    article.get("title") or "Untitled article"
                ),
                "summary": _truncate_text(
                    article.get("summary")
                    or article.get("description")
                    or "No summary available."
                ),
                "sentiment": _truncate_text(article.get("sentiment") or "unknown", 40),
                "importance_score": _format_importance(article.get("importance_score")),
                "source": _truncate_text(article.get("source") or "Unknown source", 120),
                "url": _truncate_text(article.get("url") or "No URL available.", 300),
            }
        )
    return context


def _truncate_text(text: Any, max_chars: int = EVIDENCE_TEXT_LIMIT) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


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
    language: str = "en",
) -> str:
    if "sources used" in answer.lower() or "sources utilisées" in answer.lower():
        return answer

    citations = _citation_labels(retrieved_articles)
    source_lines = _source_lines(retrieved_articles, citations)
    if not source_lines:
        return answer
    heading = "Sources utilisées" if language == "fr" else "Sources used"

    return "\n".join(
        [
            answer.rstrip(),
            "",
            f"### {heading}",
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


def _is_greeting_or_casual(normalized_question: str) -> bool:
    cleaned = normalized_question.strip(" ?.!,")
    first_word = cleaned.replace(",", " ").replace("!", " ").split(maxsplit=1)[0]
    return (
        first_word in CONVERSATION_STARTERS
        or any(phrase in cleaned for phrase in CASUAL_CONVERSATION_PHRASES)
    )


def _is_data_query(normalized_question: str) -> bool:
    words = set(normalized_question.replace("?", " ").replace(",", " ").split())
    return any(term in words or term in normalized_question for term in DATA_QUERY_TERMS)


def _normalize_question(question: str) -> str:
    return (
        question.strip()
        .lower()
        .replace("’", "'")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ç", "c")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
    )


def _is_temporal_follow_up(normalized_question: str) -> bool:
    cleaned = normalized_question.strip(" ?.!,")
    return any(cleaned == term for term in TEMPORAL_FOLLOW_UP_TERMS)


def _format_importance(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "unknown"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
