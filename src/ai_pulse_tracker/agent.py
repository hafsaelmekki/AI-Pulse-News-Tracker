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
You are AI Pulse, an AI dashboard analyst.

Your role is to analyze the dashboard data produced by AI Pulse.
You help users understand AI trends, sentiment evolution, important articles,
most mentioned companies, sources, risks, opportunities, and weak signals.

Rules:
- Use dashboard_context first.
- Use retrieved articles only as evidence or examples.
- Stay strictly inside AI Pulse dashboard data and retrieved AI Pulse articles.
- If the user asks about an unrelated topic, politely say you can only analyze AI Pulse data.
- Do not invent numbers.
- Do not invent articles.
- If the dashboard data is insufficient, say it clearly.
- Stay inside the AI / Generative AI scope.
- Ignore unrelated articles.
- Do not behave like a search engine.
- Do not simply list article titles.
- Give analytical insights.
- Mention relevant numbers from the dashboard.
- Answer in the configured assistant language from the prompt.
- Never confuse sentiment label with importance_score.
""".strip()

LLM_TIMEOUT_SECONDS = 20
LLM_MAX_RETRIES = 2
LLM_RETRY_BASE_SECONDS = 0.75
MAX_LLM_PAYLOAD_KB = 28.0
EVIDENCE_TEXT_LIMIT = 220

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
    normalized = question.strip().lower()
    if not normalized:
        return True

    first_word = normalized.replace(",", " ").replace("!", " ").split(maxsplit=1)[0]
    if first_word in CONVERSATION_STARTERS:
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
    normalized = question.strip().lower()
    first_word = normalized.replace(",", " ").replace("!", " ").split(maxsplit=1)[0]
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
    if not normalized or first_word in CONVERSATION_STARTERS:
        return _localized(
            target_language,
            (
                "Hey Hafsa. I am your AI Pulse assistant. "
                "I can analyze AI trends, companies, sources, sentiment, "
                "article importance, and dashboard weak signals. "
                "Choose a question below or ask your own."
            ),
            (
                "Hey Hafsa. Je suis ton assistant AI Pulse. "
                "Je peux analyser les tendances IA, les companies, les sources, "
                "le sentiment, l'importance des articles et les signaux faibles du dashboard. "
                "Choisis une question ci-dessous ou pose ta propre question."
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
    deterministic_answer = _deterministic_dashboard_answer(
        question,
        dashboard_context,
        intent=intent,
        language=language,
    )
    if deterministic_answer:
        return deterministic_answer

    messages = build_dashboard_llm_messages(
        question,
        dashboard_context,
        _article_context(retrieved_articles),
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
    return _ensure_source_appendix(llm_answer, retrieved_articles)


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

    header = _localized(
        language,
        "Here are the most important articles mentioning the main companies in the dashboard:",
        "Voici les articles les plus importants qui mentionnent les principales companies du dashboard :",
    )
    lines = [header, ""]
    for index, article in enumerate(matches[:5], start=1):
        title = str(article.get("title", "Untitled article")).strip() or "Untitled article"
        lines.extend(
            [
                f"{index}. {title}",
                f"- Company: {article.get('matched_company', 'unknown')}",
                f"- Source: {article.get('source', 'Unknown source')}",
                f"- Sentiment: {article.get('sentiment', 'unknown')}",
                f"- Importance: {_format_importance(article.get('importance_score'))}/100",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_dashboard_llm_messages(
    question: str,
    dashboard_context: Mapping[str, Any],
    article_context: list[Mapping[str, Any]],
    *,
    intent: str,
    language: str | None = None,
) -> list[dict[str, str]]:
    payload = _protected_dashboard_payload(
        question=question,
        dashboard_context=dashboard_context,
        article_context=article_context,
        intent=intent,
    )
    target_language = language or _answer_language(question)
    language_name = "French" if target_language == "fr" else "English"
    context_json = json.dumps(
        payload["dashboard_context"],
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    evidence_json = json.dumps(
        payload["retrieved_evidence"],
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
            "dashboard_context:",
            context_json,
            "",
            "Retrieved article evidence:",
            evidence_json,
            "",
            f"User question: {question}",
            "",
            (
                "Answer as an AI Pulse dashboard analyst. Start from dashboard_context "
                "numbers, then use retrieved evidence only to illustrate the analysis. "
                f"Write the entire answer in {language_name}."
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
        return deterministic
    if intent == "weak_signals":
        return _answer_weak_signals(dashboard_context, language, error)
    if intent == "trends":
        return _answer_trends(dashboard_context, language, error)
    return _answer_general_summary(dashboard_context, language, error)


def _answer_companies(dashboard_context: Mapping[str, Any], language: str) -> str:
    companies = _as_records(dashboard_context.get("top_companies"))[:10]
    if not companies:
        return _localized(
            language,
            "I do not have enough AI Pulse company data to answer this.",
            "Je n’ai pas assez de données company dans AI Pulse pour répondre.",
        )
    lines = [
        _localized(
            language,
            "The most mentioned companies in the current AI Pulse dashboard are:",
            "Les companies les plus mentionnées dans le dashboard AI Pulse sont :",
        )
    ]
    for company in companies:
        name = company.get("Company") or company.get("company") or "Unknown"
        articles = company.get("Articles") or company.get("articles") or 0
        avg_importance = company.get("Avg importance") or company.get("avg_importance") or "unknown"
        sentiment = company.get("Dominant sentiment") or company.get("dominant_sentiment") or "unknown"
        lines.append(
            f"- {name}: {articles} articles, average importance {avg_importance}, dominant sentiment {sentiment}."
        )
    strongest = companies[0]
    strongest_name = strongest.get("Company") or strongest.get("company") or "the top company"
    lines.append(
        _localized(
            language,
            f"{strongest_name} is the strongest company signal because it has the highest mention count in the dashboard.",
            f"{strongest_name} est le signal company le plus fort car il a le plus grand nombre de mentions dans le dashboard.",
        )
    )
    return "\n".join(lines)


def _answer_sources(dashboard_context: Mapping[str, Any], language: str) -> str:
    sources = _as_records(dashboard_context.get("top_sources"))[:10]
    if not sources:
        return _localized(
            language,
            "I do not have enough AI Pulse source data to answer this.",
            "Je n’ai pas assez de données sources dans AI Pulse pour répondre.",
        )
    lines = [
        _localized(
            language,
            "The most active AI Pulse sources are:",
            "Les sources AI Pulse les plus actives sont :",
        )
    ]
    for source in sources:
        name = source.get("Source") or source.get("source") or "Unknown"
        articles = source.get("articles") or source.get("Articles") or 0
        avg_importance = source.get("Avg importance") or source.get("avg_importance") or "unknown"
        lines.append(f"- {name}: {articles} articles, average importance {avg_importance}.")
    return "\n".join(lines)


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
    lines = [
        _localized(
            language,
            f"The dominant AI Pulse sentiment is {dominant}. Distribution:",
            f"Le sentiment dominant dans AI Pulse est {dominant}. Distribution :",
        )
    ]
    for item in distribution:
        lines.append(f"- {item.get('sentiment', 'unknown')}: {item.get('articles', 0)} articles")
    if sentiment_by_day:
        lines.append(
            _localized(
                language,
                "Recent daily sentiment percentages are available in the dashboard context for the last days.",
                "Les pourcentages quotidiens récents sont disponibles dans le contexte dashboard.",
            )
        )
    return "\n".join(lines)


def _answer_importance(dashboard_context: Mapping[str, Any], language: str) -> str:
    articles = _as_records(dashboard_context.get("top_important_articles"))[:5]
    average_importance = dashboard_context.get("average_importance_score", 0.0)
    if not articles:
        return _localized(
            language,
            "I do not have enough AI Pulse importance data to answer this.",
            "Je n’ai pas assez de données d’importance AI Pulse pour répondre.",
        )
    lines = [
        _localized(
            language,
            f"The average AI Pulse importance score is {average_importance}. Top important articles:",
            f"Le score d’importance moyen AI Pulse est {average_importance}. Articles les plus importants :",
        )
    ]
    for index, article in enumerate(articles, start=1):
        lines.extend(
            [
                f"{index}. {article.get('title', 'Untitled article')}",
                f"- Source: {article.get('source', 'Unknown source')}",
                f"- Sentiment: {article.get('sentiment', 'unknown')}",
                f"- Importance: {_format_importance(article.get('importance_score'))}/100",
            ]
        )
    return "\n".join(lines)


def _answer_weak_signals(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    signals = _as_records(dashboard_context.get("weak_signals"))[:6]
    if not signals:
        return _fallback_notice(
            language,
            "No weak signals are currently visible in the compact AI Pulse dashboard context.",
            "Aucun signal faible n’est visible dans le contexte compact AI Pulse.",
            error,
        )
    lines = [
        _fallback_notice(
            language,
            "LLM synthesis is unavailable, so here are deterministic weak signals from AI Pulse:",
            "La synthèse LLM est indisponible, voici les signaux faibles déterministes depuis AI Pulse :",
            error,
        )
    ]
    for signal in signals:
        topic = signal.get("Topic") or signal.get("topic") or "Unknown"
        articles = signal.get("Articles") or signal.get("articles") or 0
        avg_importance = signal.get("Avg importance") or signal.get("avg_importance") or "unknown"
        lines.append(f"- {topic}: {articles} articles, average importance {avg_importance}.")
    return "\n".join(lines)


def _answer_trends(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    topics = _as_records(dashboard_context.get("top_topics"))[:5]
    if not topics:
        return _answer_general_summary(dashboard_context, language, error)
    lines = [
        _fallback_notice(
            language,
            "LLM synthesis is unavailable, so here are deterministic AI Pulse trends:",
            "La synthèse LLM est indisponible, voici les tendances déterministes AI Pulse :",
            error,
        )
    ]
    for topic in topics:
        name = topic.get("topic") or topic.get("Topic") or "Unknown"
        articles = topic.get("articles") or topic.get("Articles") or 0
        avg_importance = topic.get("avg_importance") or topic.get("Avg importance") or "unknown"
        lines.append(f"- {name}: {articles} articles, average importance {avg_importance}.")
    return "\n".join(lines)


def _answer_general_summary(
    dashboard_context: Mapping[str, Any],
    language: str,
    error: str,
) -> str:
    total_articles = dashboard_context.get("total_articles", 0)
    date_range = dashboard_context.get("date_range", {})
    dominant = dashboard_context.get("dominant_sentiment", "unknown")
    average_importance = dashboard_context.get("average_importance_score", 0.0)
    return _fallback_notice(
        language,
        (
            f"LLM synthesis is unavailable, but AI Pulse currently contains {total_articles} articles "
            f"from {date_range.get('start')} to {date_range.get('end')}. "
            f"Dominant sentiment: {dominant}. Average importance: {average_importance}."
        ),
        (
            f"La synthèse LLM est indisponible, mais AI Pulse contient actuellement {total_articles} articles "
            f"du {date_range.get('start')} au {date_range.get('end')}. "
            f"Sentiment dominant : {dominant}. Importance moyenne : {average_importance}."
        ),
        error,
    )


def _fallback_notice(language: str, english: str, french: str, error: str) -> str:
    notice = _localized(language, english, french)
    if "rate-limited" in error.lower() or "429" in error:
        return notice
    return notice


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
        "bonjour",
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
    article_context: list[Mapping[str, Any]],
    intent: str,
) -> dict[str, Any]:
    payload = {
        "intent": intent,
        "question": question,
        "dashboard_context": dict(dashboard_context),
        "retrieved_evidence": [_compact_evidence_article(article) for article in article_context[:3]],
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
        _without_keys(dict(article), {"url", "description", "content"})
        for article in list(payload.get("retrieved_evidence", []))[:2]
    ]
    return {
        "intent": payload.get("intent"),
        "question": payload.get("question"),
        "dashboard_context": reduced_context,
        "retrieved_evidence": reduced_evidence,
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
        "question": payload.get("question"),
        "dashboard_context": minimal_context,
        "retrieved_evidence": [],
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


def _normalize_question(question: str) -> str:
    return (
        question.strip()
        .lower()
        .replace("’", "'")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
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
