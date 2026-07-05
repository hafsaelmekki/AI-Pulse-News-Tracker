from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from .agent import answer_conversation, answer_question, is_conversation_prompt
from .config import load_settings
from .models import UpsertResult
from .pipeline import NewsAnalyzerPipeline
from .retrieval import search_articles
from .trends import count_keywords, normalize_keywords

st.set_page_config(page_title="AI Pulse Tracker", layout="wide")
PIPELINE_RESOURCE_KEY = "pipeline-v2"
TRACKED_TOPICS = {
    "AI Agents": ("ai agent", "ai agents", "agentic", "agents", "agent"),
    "RAG": ("rag", "retrieval augmented", "retrieval"),
    "OpenAI": ("openai", "chatgpt", "gpt-"),
    "Microsoft": ("microsoft", "azure", "copilot"),
    "Google": ("google", "gemini", "deepmind"),
    "Azure": ("azure",),
    "Regulation": ("regulation", "regulatory", "régulation", "reglementation"),
    "Coding Agents": ("coding agent", "code agent", "developer agent", "github copilot"),
}
TRACKED_COMPANIES = {
    "OpenAI": ("openai", "chatgpt", "gpt-"),
    "Google": ("google", "gemini", "deepmind"),
    "Microsoft": ("microsoft", "azure", "copilot"),
    "Meta": ("meta", "llama"),
    "Anthropic": ("anthropic", "claude"),
    "Mistral AI": ("mistral", "mistral ai"),
    "Nvidia": ("nvidia",),
    "Apple": ("apple",),
}


@st.cache_resource(show_spinner=False)
def get_pipeline(_cache_buster: str) -> NewsAnalyzerPipeline:
    settings = load_settings()
    return NewsAnalyzerPipeline(settings)


@st.cache_data(ttl=600, show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    rows = get_pipeline(PIPELINE_RESOURCE_KEY).load_dashboard_rows()
    return pd.DataFrame(rows)


def _ensure_upsert_result(result: UpsertResult | list[str]) -> UpsertResult:
    if isinstance(result, UpsertResult):
        return result
    if hasattr(result, "ids") and hasattr(result, "created"):
        return UpsertResult(
            ids=list(getattr(result, "ids", [])),
            created=int(getattr(result, "created", 0)),
            updated=int(getattr(result, "updated", 0)),
        )
    return UpsertResult(ids=list(result), created=len(result), updated=0)


def _parse_since_text(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        st.error("Invalid ISO8601 datetime. Example: 2024-04-02T09:30:00Z")
        return None


def _time_filters() -> dict[str, timedelta | None]:
    return {
        "Last 5 minutes": timedelta(minutes=5),
        "Last 1 hour": timedelta(hours=1),
        "Last 24 hours": timedelta(days=1),
        "Last 7 days": timedelta(days=7),
        "Last 30 days": timedelta(days=30),
        "All time": None,
    }


def _apply_filters(
    df: pd.DataFrame,
    sentiments: list[str],
    sources: list[str],
    min_importance: float,
    date_range: tuple[object, object] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered["date_dt"] = pd.to_datetime(
        filtered["date"], utc=True, errors="coerce")
    if date_range is not None:
        start_date, end_date = date_range
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        filtered = filtered[
            (filtered["date_dt"] >= start_dt) & (filtered["date_dt"] < end_dt)
        ]

    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]

    if sources:
        filtered = filtered[filtered["source"].isin(sources)]

    filtered = filtered[filtered["importance_score"] >= min_importance]

    return filtered.drop(columns=["date_dt"])


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "source_name" in prepared.columns:
        prepared["source"] = prepared["source_name"].fillna(prepared["source"])
    if "description" not in prepared.columns:
        prepared["description"] = ""
    if "summary" not in prepared.columns:
        prepared["summary"] = ""
    prepared["summary"] = prepared["summary"].fillna("").astype(str)
    missing_summary = prepared["summary"].str.strip() == ""
    prepared.loc[missing_summary, "summary"] = prepared.loc[
        missing_summary,
        "description",
    ].fillna("")
    missing_summary = prepared["summary"].str.strip() == ""
    prepared.loc[missing_summary, "summary"] = prepared.loc[
        missing_summary,
        "title",
    ].fillna("")
    if "keywords" not in prepared.columns:
        prepared["keywords"] = [[] for _ in range(len(prepared))]
    prepared["keywords"] = prepared["keywords"].apply(normalize_keywords)
    if "importance_score" not in prepared.columns:
        prepared["importance_score"] = 0.0
    prepared["importance_score"] = pd.to_numeric(
        prepared["importance_score"],
        errors="coerce",
    ).fillna(0.0)
    prepared["topics"] = prepared.apply(_detect_topics, axis=1)
    prepared["topic"] = prepared["topics"].apply(
        lambda topics: topics[0] if topics else "Uncategorized"
    )
    return prepared


def _keyword_dataframe(df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    return pd.DataFrame(count_keywords(df.to_dict("records"), limit=limit))


def _detect_topics(row: pd.Series) -> list[str]:
    text = _row_text(row)
    topics = [
        topic
        for topic, aliases in TRACKED_TOPICS.items()
        if any(alias in text for alias in aliases)
    ]
    if topics:
        return topics

    keywords = normalize_keywords(row.get("keywords"))
    return [_title_case_topic(keyword) for keyword in keywords[:3]]


def _detect_companies(row: pd.Series) -> list[str]:
    text = _row_text(row)
    return [
        company
        for company, aliases in TRACKED_COMPANIES.items()
        if any(alias in text for alias in aliases)
    ]


def _row_text(row: pd.Series) -> str:
    keywords = " ".join(normalize_keywords(row.get("keywords")))
    return " ".join(
        [
            str(row.get("title", "")),
            str(row.get("summary", "")),
            str(row.get("description", "")),
            keywords,
        ]
    ).lower()


def _title_case_topic(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def _confidence_score(row: object, key: str) -> float:
    if isinstance(row, dict):
        return float(row.get(key, 0.0) or 0.0)
    return 0.0


def _add_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    confidence = scored.get(
        "confidence",
        pd.Series([{} for _ in range(len(scored))], index=scored.index),
    )
    scored["pos_score"] = confidence.apply(
        lambda row: _confidence_score(row, "pos")
    )
    scored["neu_score"] = confidence.apply(
        lambda row: _confidence_score(row, "neu")
    )
    scored["neg_score"] = confidence.apply(
        lambda row: _confidence_score(row, "neg")
    )
    scored["sentiment_score"] = scored["pos_score"] - scored["neg_score"]
    return scored


def _render_project_menu() -> str:
    st.sidebar.header("Project Menu")
    return st.sidebar.radio(
        "Sections",
        ["Dashboard", "Assistant"],
        key="project-section",
    )


def _render_kpi_cards(df: pd.DataFrame) -> None:
    st.subheader("Dashboard KPIs")

    date_series = pd.to_datetime(df["date"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    today_count = int((date_series.dt.date == now.date()).sum())
    week_count = int((date_series >= now - timedelta(days=7)).sum())
    sentiment_label, sentiment_delta = _average_sentiment(df)
    positive_pct = _sentiment_percentage(df, "positive")
    negative_pct = _sentiment_percentage(df, "negative")
    top_article = _top_article_label(df)
    top_source = _top_value(df, "source")
    dominant_topic = _dominant_topic(df)

    first_row = st.columns(4)
    first_row[0].metric("Total articles", int(len(df)))
    first_row[1].metric("Articles today", today_count)
    first_row[2].metric("Articles this week", week_count)
    first_row[3].metric("Average sentiment", sentiment_label, sentiment_delta)

    second_row = st.columns(4)
    second_row[0].metric("Positive articles", f"{positive_pct:.0f}%")
    second_row[1].metric("Negative articles", f"{negative_pct:.0f}%")
    second_row[2].metric("Most active source", top_source)
    second_row[3].metric("Dominant topic", dominant_topic)

    st.metric("Most important article", top_article)


def _average_sentiment(df: pd.DataFrame) -> tuple[str, str]:
    sentiment_scores = (
        df["sentiment"]
        .fillna("")
        .astype(str)
        .str.lower()
        .map({"positive": 1.0, "neutral": 0.0, "negative": -1.0})
        .dropna()
    )
    if sentiment_scores.empty:
        return "N/A", "no sentiment"

    average = float(sentiment_scores.mean())
    if average >= 0.25:
        label = "Positive"
    elif average <= -0.25:
        label = "Negative"
    else:
        label = "Neutral"
    return label, f"{average:+.2f}"


def _sentiment_percentage(df: pd.DataFrame, sentiment: str) -> float:
    if df.empty:
        return 0.0
    sentiments = df["sentiment"].fillna("").astype(str).str.lower()
    return float(sentiments.eq(sentiment).mean() * 100)


def _top_article_label(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    article = df.sort_values("importance_score", ascending=False).iloc[0]
    title = str(article.get("title", "")).strip() or "Untitled article"
    score = float(article.get("importance_score", 0.0) or 0.0)
    return f"{_shorten_text(title, 80)} ({score:.1f})"


def _top_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "N/A"
    counts = df[column].fillna("").astype(str).str.strip()
    counts = counts[counts != ""].value_counts()
    if counts.empty:
        return "N/A"
    return _shorten_text(str(counts.index[0]), 40)


def _dominant_topic(df: pd.DataFrame) -> str:
    keyword_df = _keyword_dataframe(df, limit=1)
    if keyword_df.empty:
        return "N/A"
    return _shorten_text(str(keyword_df.iloc[0]["keyword"]), 40)


def _shorten_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _render_trends(df: pd.DataFrame) -> None:
    st.subheader("Trends & Topics")

    topic_col, evolution_col = st.columns(2)
    topic_df = _topic_counts_dataframe(df)
    with topic_col:
        st.markdown("**Top keywords / topics**")
        if topic_df.empty:
            st.info("No topics available yet.")
        else:
            st.plotly_chart(
                px.bar(
                    topic_df.head(12),
                    x="articles",
                    y="topic",
                    orientation="h",
                    text="articles",
                ).update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Articles",
                    yaxis_title="Topic",
                ),
                use_container_width=True,
            )

    with evolution_col:
        st.markdown("**Topic evolution over time**")
        evolution_df = _topic_evolution_dataframe(
            df, topic_df.head(6)["topic"].tolist())
        if evolution_df.empty:
            st.info("Not enough dated topic data yet.")
        else:
            st.plotly_chart(
                px.line(
                    evolution_df,
                    x="date",
                    y="articles",
                    color="topic",
                    markers=True,
                ).update_layout(
                    xaxis_title="Date",
                    yaxis_title="Articles",
                ),
                use_container_width=True,
            )

    weak_df = _weak_signals_dataframe(df)
    st.markdown("**Weak signals to watch**")
    if weak_df.empty:
        st.info("No weak signal detected yet.")
    else:
        st.dataframe(weak_df, use_container_width=True, hide_index=True)


def _topic_occurrences_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, article in df.iterrows():
        topics = article.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        article_date = pd.to_datetime(
            article.get("date"), utc=True, errors="coerce")
        for topic in topics:
            rows.append(
                {
                    "date": article_date.date() if not pd.isna(article_date) else None,
                    "topic": topic,
                    "importance_score": float(article.get("importance_score", 0.0) or 0.0),
                    "sentiment": str(article.get("sentiment", "")).lower(),
                    "title": str(article.get("title", "")),
                }
            )
    return pd.DataFrame(rows)


def _topic_counts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    occurrences = _topic_occurrences_dataframe(df)
    if occurrences.empty:
        return pd.DataFrame(columns=["topic", "articles", "avg_importance"])
    return (
        occurrences.groupby("topic", as_index=False)
        .agg(
            articles=("title", "count"),
            avg_importance=("importance_score", "mean"),
        )
        .sort_values(["articles", "avg_importance"], ascending=False)
    )


def _topic_evolution_dataframe(df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    occurrences = _topic_occurrences_dataframe(df)
    if occurrences.empty or not topics:
        return pd.DataFrame(columns=["date", "topic", "articles"])
    occurrences = occurrences[
        occurrences["topic"].isin(topics) & occurrences["date"].notna()
    ]
    if occurrences.empty:
        return pd.DataFrame(columns=["date", "topic", "articles"])
    return (
        occurrences.groupby(["date", "topic"], as_index=False)
        .agg(articles=("title", "count"))
        .sort_values("date")
    )


def _weak_signals_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    topic_df = _topic_counts_dataframe(df)
    if topic_df.empty:
        return pd.DataFrame(columns=["Topic", "Articles", "Avg importance", "Signal"])

    weak_df = topic_df[
        (topic_df["articles"] <= 3) & (topic_df["avg_importance"] >= 70)
    ].copy()
    if weak_df.empty:
        return pd.DataFrame(columns=["Topic", "Articles", "Avg importance", "Signal"])

    weak_df = weak_df.sort_values("avg_importance", ascending=False).head(6)
    return pd.DataFrame(
        {
            "Topic": weak_df["topic"],
            "Articles": weak_df["articles"].astype(int),
            "Avg importance": weak_df["avg_importance"].map(lambda score: f"{score:.1f}"),
            "Signal": "Low frequency, high importance",
        }
    )


def _render_sentiment_analysis(df: pd.DataFrame) -> None:
    st.subheader("Sentiment Analysis")

    sentiment_df = _sentiment_distribution_dataframe(df)
    daily_score_df = _daily_sentiment_score_dataframe(df)

    distribution_col, evolution_col = st.columns(2)
    with distribution_col:
        st.markdown("**Sentiment distribution**")
        if sentiment_df.empty:
            st.info("No sentiment data available yet.")
        else:
            st.plotly_chart(
                px.pie(
                    sentiment_df,
                    names="sentiment",
                    values="articles",
                    hole=0.45,
                    color="sentiment",
                    color_discrete_map={
                        "positive": "#00CC96",
                        "neutral": "#636EFA",
                        "negative": "#EF553B",
                    },
                ),
                use_container_width=True,
            )

    with evolution_col:
        st.markdown("**Average sentiment over time**")
        if daily_score_df.empty:
            st.info("Not enough dated articles to plot sentiment evolution.")
        else:
            st.plotly_chart(
                px.line(
                    daily_score_df,
                    x="date",
                    y="average_sentiment",
                    markers=True,
                    range_y=[-1, 1],
                ).update_layout(
                    yaxis_title="Average sentiment (-1 negative, +1 positive)",
                    xaxis_title="Date",
                ),
                use_container_width=True,
            )

    insight = _sentiment_trend_insight(daily_score_df)
    if insight:
        st.info(insight)


def _sentiment_distribution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sentiments = df["sentiment"].fillna("").astype(str).str.lower().str.strip()
    sentiments = sentiments[sentiments != ""]
    if sentiments.empty:
        return pd.DataFrame(columns=["sentiment", "articles"])
    return (
        sentiments.value_counts()
        .rename_axis("sentiment")
        .reset_index(name="articles")
        .sort_values("articles", ascending=False)
    )


def _daily_sentiment_score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(
        trend_df["date"], utc=True, errors="coerce")
    trend_df["sentiment_score"] = (
        trend_df["sentiment"]
        .fillna("")
        .astype(str)
        .str.lower()
        .map({"positive": 1.0, "neutral": 0.0, "negative": -1.0})
    )
    trend_df = trend_df.dropna(subset=["date", "sentiment_score"])
    if trend_df.empty:
        return pd.DataFrame(columns=["date", "average_sentiment", "articles"])

    trend_df["date"] = trend_df["date"].dt.date
    return (
        trend_df.groupby("date", as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            articles=("title", "count"),
        )
        .sort_values("date")
    )


def _sentiment_trend_insight(daily_score_df: pd.DataFrame) -> str | None:
    if len(daily_score_df) < 2:
        return None

    recent_df = daily_score_df.tail(3)
    previous_score = float(recent_df.iloc[0]["average_sentiment"])
    latest_score = float(recent_df.iloc[-1]["average_sentiment"])
    delta = latest_score - previous_score

    if delta <= -0.2:
        return "Signal: sentiment has become more negative over the latest available days."
    if delta >= 0.2:
        return "Signal: sentiment has become more positive over the latest available days."
    return "Signal: sentiment is broadly stable over the latest available days."


def _render_importance_analysis(df: pd.DataFrame) -> None:
    st.subheader("Article Importance")

    top_col, distribution_col = st.columns((1.4, 1))
    with top_col:
        st.markdown("**Top 5 most important articles**")
        top_articles = _top_importance_articles(df)
        if top_articles.empty:
            st.info("No importance scores available yet.")
        else:
            st.dataframe(top_articles, use_container_width=True,
                         hide_index=True)

    with distribution_col:
        st.markdown("**Importance score distribution**")
        distribution_df = _importance_distribution_dataframe(df)
        st.plotly_chart(
            px.bar(
                distribution_df,
                x="bucket",
                y="articles",
                text="articles",
                color="bucket",
                color_discrete_sequence=px.colors.sequential.Blues[2:],
            ).update_layout(
                xaxis_title="Importance score",
                yaxis_title="Articles",
                showlegend=False,
            ),
            use_container_width=True,
        )

    st.markdown("**Sentiment vs importance**")
    st.plotly_chart(
        px.scatter(
            df,
            x="importance_score",
            y="sentiment_score",
            color="sentiment",
            hover_data=["title", "source", "date"],
            color_discrete_map={
                "positive": "#00CC96",
                "neutral": "#636EFA",
                "negative": "#EF553B",
            },
        ).update_layout(
            xaxis_title="Importance score",
            yaxis_title="Sentiment score",
        ),
        use_container_width=True,
    )


def _top_importance_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["Title", "Source", "Date",
                     "Sentiment", "Importance", "URL"]
        )

    top_df = df.sort_values("importance_score", ascending=False).head(5).copy()
    return pd.DataFrame(
        {
            "Title": top_df["title"].fillna("").astype(str),
            "Source": top_df["source"].fillna("").astype(str),
            "Date": top_df["date"].fillna("").astype(str),
            "Sentiment": top_df["sentiment"].fillna("").astype(str).str.capitalize(),
            "Importance": top_df["importance_score"].map(lambda score: f"{score:.1f}"),
            "URL": top_df["url"].fillna("").astype(str),
        }
    )


def _importance_distribution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    buckets = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    scores = pd.to_numeric(df["importance_score"], errors="coerce").fillna(0.0)
    bucket_series = pd.cut(
        scores,
        bins=[0, 20, 40, 60, 80, 100],
        labels=buckets,
        include_lowest=True,
        right=True,
    )
    counts = bucket_series.value_counts().reindex(buckets, fill_value=0)
    return pd.DataFrame({"bucket": buckets, "articles": counts.astype(int).tolist()})


def _render_sources_companies(df: pd.DataFrame) -> None:
    st.subheader("Sources & Companies")

    source_col, company_col = st.columns(2)
    source_df = _source_strategy_dataframe(df)
    with source_col:
        st.markdown("**Most active sources**")
        if source_df.empty:
            st.info("No source data available yet.")
        else:
            st.plotly_chart(
                px.bar(
                    source_df.head(10),
                    x="articles",
                    y="Source",
                    orientation="h",
                    color="Avg importance",
                    color_continuous_scale="Blues",
                ).update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Articles",
                    yaxis_title="Source",
                ),
                use_container_width=True,
            )
            st.dataframe(source_df, use_container_width=True, hide_index=True)

    company_df = _company_strategy_dataframe(df)
    with company_col:
        st.markdown("**Most mentioned companies**")
        if company_df.empty:
            st.info("No tracked company mentions detected yet.")
        else:
            st.plotly_chart(
                px.bar(
                    company_df.head(10),
                    x="Articles",
                    y="Company",
                    orientation="h",
                    color="Avg importance",
                    color_continuous_scale="Purples",
                ).update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Articles",
                    yaxis_title="Company",
                ),
                use_container_width=True,
            )
            st.dataframe(company_df, use_container_width=True, hide_index=True)


def _source_strategy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["Source", "articles", "Avg sentiment", "Avg importance"]
        )

    source_df = (
        df.groupby("source", as_index=False)
        .agg(
            articles=("title", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_importance=("importance_score", "mean"),
        )
        .sort_values("articles", ascending=False)
    )
    return pd.DataFrame(
        {
            "Source": source_df["source"],
            "articles": source_df["articles"].astype(int),
            "Avg sentiment": source_df["avg_sentiment"].map(lambda score: f"{score:+.2f}"),
            "Avg importance": source_df["avg_importance"].round(1),
        }
    )


def _company_mentions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, article in df.iterrows():
        for company in _detect_companies(article):
            rows.append(
                {
                    "company": company,
                    "title": str(article.get("title", "")),
                    "sentiment": str(article.get("sentiment", "")).lower(),
                    "importance_score": float(article.get("importance_score", 0.0) or 0.0),
                }
            )
    return pd.DataFrame(rows)


def _company_strategy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mentions = _company_mentions_dataframe(df)
    if mentions.empty:
        return pd.DataFrame(
            columns=["Company", "Articles",
                     "Dominant sentiment", "Avg importance"]
        )

    grouped = (
        mentions.groupby("company", as_index=False)
        .agg(
            articles=("title", "count"),
            avg_importance=("importance_score", "mean"),
            dominant_sentiment=("sentiment", _dominant_sentiment_value),
        )
        .sort_values(["articles", "avg_importance"], ascending=False)
    )
    return pd.DataFrame(
        {
            "Company": grouped["company"],
            "Articles": grouped["articles"].astype(int),
            "Dominant sentiment": grouped["dominant_sentiment"].str.capitalize(),
            "Avg importance": grouped["avg_importance"].round(1),
        }
    )


def _dominant_sentiment_value(values: pd.Series) -> str:
    values = values.fillna("").astype(str)
    values = values[values != ""]
    if values.empty:
        return "unknown"
    return str(values.value_counts().idxmax())


def _render_evidence(search_results: list[dict[str, object]]) -> None:
    if not search_results:
        return

    with st.expander("Retrieved article evidence", expanded=False):
        st.dataframe(
            pd.DataFrame(search_results)[
                [
                    "rank",
                    "score",
                    "vector_score",
                    "source",
                    "title",
                    "summary",
                    "matched_terms",
                    "importance_score",
                    "url",
                ]
            ],
            use_container_width=True,
        )


def _render_ai_assistant(df: pd.DataFrame) -> None:
    st.subheader("AI Trend Assistant")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    if st.button("Clear chat", key="clear-agent-chat"):
        st.session_state.agent_messages = []

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_evidence(message.get("evidence", []))

    prompt = st.chat_input(
        "Ask about AI trends, companies, RAG, agents...",
        key="ai-agent-chat-input",
    )
    if not prompt:
        return

    if is_conversation_prompt(prompt):
        search_results = []
        answer = answer_conversation(prompt)
    else:
        search_results = search_articles(
            df.to_dict("records"),
            prompt,
            limit=5,
        )
        answer = answer_question(prompt, search_results)

    user_message = {"role": "user", "content": prompt}
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "evidence": search_results,
    }
    st.session_state.agent_messages.extend([user_message, assistant_message])

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_evidence(search_results)


def _render_articles(df: pd.DataFrame) -> None:
    st.subheader("Article Explorer")

    explorer_df = pd.DataFrame(
        {
            "Date": df["date"].fillna("").astype(str),
            "Title": df["title"].fillna("").astype(str),
            "Source": df["source"].fillna("").astype(str),
            "Topic": df["topic"].fillna("").astype(str),
            "Sentiment": df["sentiment"].fillna("").astype(str).str.capitalize(),
            "Sentiment score": df["sentiment_score"].round(2),
            "Importance score": df["importance_score"].round(1),
            "URL": df["url"].fillna("").astype(str),
            "Summary": df["summary"].fillna("").astype(str),
        }
    )
    st.dataframe(explorer_df, use_container_width=True, hide_index=True)


def _render_dashboard_header(df: pd.DataFrame) -> None:
    date_series = pd.to_datetime(
        df["date"], utc=True, errors="coerce").dropna()
    if date_series.empty:
        st.caption("Last update: unknown")
        return
    last_update = date_series.max().strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Last update: {last_update}")


def _date_bounds(df: pd.DataFrame) -> tuple[object, object]:
    date_series = pd.to_datetime(
        df["date"], utc=True, errors="coerce").dropna()
    if date_series.empty:
        today = datetime.now(timezone.utc).date()
        return today, today
    return date_series.min().date(), date_series.max().date()


def _normalize_date_range(value: object) -> tuple[object, object] | None:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    return None


def _render_dashboard_view(df: pd.DataFrame) -> None:
    _render_dashboard_header(df)

    st.subheader("Global Filters")
    sentiments_available = sorted(df["sentiment"].dropna().unique().tolist())
    sources_available = sorted(df["source"].fillna(
        "").astype(str).unique().tolist())
    sources_available = [source for source in sources_available if source]
    max_importance = float(df["importance_score"].max()
                           ) if not df.empty else 0.0
    min_date, max_date = _date_bounds(df)

    col_date, col_sentiment, col_source, col_importance = st.columns(
        (1.2, 1, 1, 1))
    selected_date_range = col_date.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    selected_sentiments = col_sentiment.multiselect(
        "Sentiment",
        sentiments_available,
        default=sentiments_available,
    )
    selected_sources = col_source.multiselect(
        "Source",
        sources_available,
        default=sources_available,
    )
    min_importance = col_importance.slider(
        "Min importance",
        min_value=0.0,
        max_value=max(100.0, max_importance),
        value=0.0,
        step=5.0,
    )

    filtered_df = _apply_filters(
        df,
        selected_sentiments,
        selected_sources,
        min_importance,
        _normalize_date_range(selected_date_range),
    )

    if filtered_df.empty:
        st.info("No articles match the selected filters.")
        return

    filtered_df = _add_confidence_scores(filtered_df)
    _render_kpi_cards(filtered_df)
    st.divider()
    _render_sentiment_analysis(filtered_df)
    st.divider()
    _render_importance_analysis(filtered_df)
    st.divider()
    _render_trends(filtered_df)
    st.divider()
    _render_sources_companies(filtered_df)
    st.divider()
    _render_articles(filtered_df)


def render_dashboard() -> None:
    st.title("AI Pulse - Azure Sentiment Monitor")
    st.caption("Latest French-language AI coverage scored via Azure AI Language")

    selected_section = _render_project_menu()

    df = load_dataframe()

    if df.empty:
        if selected_section == "Assistant":
            _render_ai_assistant(pd.DataFrame())
        else:
            st.warning(
                "No documents found in Cosmos DB. Run the ingestion pipeline first.")
        return

    df = _prepare_dataframe(df)

    if selected_section == "Dashboard":
        _render_dashboard_view(df)
    elif selected_section == "Assistant":
        _render_ai_assistant(df)


if __name__ == "__main__":
    render_dashboard()
