from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .agent import (
    answer_conversation,
    answer_dashboard_question,
    detect_assistant_language,
    detect_dashboard_intent,
    detect_language_change_request,
    is_conversation_prompt,
)
from .config import load_settings
from .models import UpsertResult
from .pipeline import NewsAnalyzerPipeline
from .relevance import ai_relevance_reason
from .retrieval import search_articles
from .trends import count_keywords, normalize_keywords

APP_PAGE_TITLE = "AI Pulse - Azure Sentiment Monitor"

st.set_page_config(
    page_title=APP_PAGE_TITLE,
    page_icon="📊",
    layout="wide",
    menu_items={"About": APP_PAGE_TITLE},
)
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
SENTIMENT_COLORS = {
    "positive": "#4ECB71",
    "neutral": "#F28E2B",
    "negative": "#F05A5A",
}
SENTIMENT_LABELS = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
}
DONUT_COLORS = [
    "#0891B2",
    "#7C3AED",
    "#2563EB",
    "#DB2777",
    "#14B8A6",
    "#94A3B8",
]
ASSISTANT_DEFAULT_SUGGESTIONS = [
    "What are the strongest AI Pulse trends right now?",
    "Which companies are most visible in AI Pulse?",
    "What changed in AI Pulse sentiment over time?",
]
ASSISTANT_DEFAULT_SUGGESTIONS_BY_LANGUAGE = {
    "en": ASSISTANT_DEFAULT_SUGGESTIONS,
    "fr": [
        "Quelles sont les tendances AI Pulse les plus fortes ?",
        "Quelles companies sont les plus visibles dans AI Pulse ?",
        "Comment évolue le sentiment AI Pulse dans le temps ?",
    ],
}
CONTEXT_TEXT_LIMIT = 220
CONTEXT_SENTIMENT_DAYS = 14


@st.cache_resource(show_spinner=False)
def get_pipeline(_cache_buster: str) -> NewsAnalyzerPipeline:
    settings = load_settings()
    return NewsAnalyzerPipeline(settings)


@st.cache_data(ttl=600, show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    rows = get_pipeline(PIPELINE_RESOURCE_KEY).load_dashboard_rows()
    return pd.DataFrame(rows)


def _run_dashboard_update(last_update: datetime | None) -> None:
    after = last_update
    if after is None:
        after = datetime.now(timezone.utc) - timedelta(days=30)

    try:
        with st.spinner("Checking for new articles..."):
            result = _ensure_upsert_result(
                get_pipeline(PIPELINE_RESOURCE_KEY).run(
                    after=after, incremental=False
                )
            )
    except Exception as exc:
        st.error(f"Update failed: {exc}")
        return

    load_dataframe.clear()
    if result.created or result.updated:
        st.session_state["dashboard_update_message"] = (
            "success",
            f"Update complete: {result.created} new articles, "
            f"{result.updated} updated.",
        )
    else:
        st.session_state["dashboard_update_message"] = (
            "info",
            "No new articles found.",
        )
    st.rerun()


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


def _date_filter_options() -> dict[str, timedelta | None]:
    return {
        "Last week": timedelta(days=7),
        "Last month": timedelta(days=30),
        "Last year": timedelta(days=365),
        "Range": None,
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
    relevance_reasons = prepared.apply(ai_relevance_reason, axis=1)
    existing_relevance = prepared.get(
        "ai_relevance",
        pd.Series([False for _ in range(len(prepared))], index=prepared.index),
    ).fillna(False).astype(bool)
    computed_relevance = relevance_reasons.astype(bool)
    prepared = prepared[existing_relevance | computed_relevance].copy()
    if prepared.empty:
        return prepared
    if "ai_relevance_reason" not in prepared.columns:
        prepared["ai_relevance_reason"] = ""
    prepared["ai_relevance_reason"] = prepared["ai_relevance_reason"].fillna("").astype(str)
    missing_relevance_reason = prepared["ai_relevance_reason"].str.strip() == ""
    prepared.loc[missing_relevance_reason, "ai_relevance_reason"] = relevance_reasons.loc[
        prepared.index
    ]
    prepared["ai_relevance"] = True
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
    return [
        topic
        for topic, aliases in TRACKED_TOPICS.items()
        if any(alias in text for alias in aliases)
    ]


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
    if "project-section" not in st.session_state:
        st.session_state["project-section"] = "Dashboard"

    for section in ("Dashboard", "Assistant"):
        is_active = st.session_state["project-section"] == section
        st.sidebar.button(
            section,
            key=f"project-menu-{section.lower()}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=_set_project_section,
            args=(section,),
        )
    return str(st.session_state["project-section"])


def _set_project_section(section: str) -> None:
    st.session_state["project-section"] = section


def _render_kpi_cards(df: pd.DataFrame) -> None:
    st.subheader("Dashboard KPIs")

    date_series = pd.to_datetime(df["date"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    today_count = int((date_series.dt.date == now.date()).sum())
    week_count = int((date_series >= now - timedelta(days=7)).sum())
    sentiment_label, sentiment_delta = _average_sentiment(df)
    positive_pct = _sentiment_percentage(df, "positive")
    negative_pct = _sentiment_percentage(df, "negative")
    top_source = _top_value(df, "source")
    dominant_topic = _dominant_topic(df)

    first_row = st.columns(4)
    _render_kpi_card(first_row[0], "Total articles", int(len(df)), "#2563EB")
    _render_kpi_card(first_row[1], "Articles today", today_count, "#0891B2")
    _render_kpi_card(first_row[2], "Articles this week", week_count, "#7C3AED")
    _render_kpi_card(
        first_row[3],
        "Average sentiment",
        sentiment_label,
        "#16A34A" if sentiment_label == "Positive" else "#F97316",
        sentiment_delta,
    )

    second_row = st.columns(4)
    _render_kpi_card(
        second_row[0],
        "Positive articles",
        f"{positive_pct:.0f}%",
        "#22C55E",
    )
    _render_kpi_card(
        second_row[1],
        "Negative articles",
        f"{negative_pct:.0f}%",
        "#EF4444",
    )
    _render_kpi_card(second_row[2], "Most active source", top_source, "#F59E0B")
    _render_kpi_card(second_row[3], "Dominant topic", dominant_topic, "#DB2777")


def _render_kpi_card(
    container: object,
    label: str,
    value: object,
    color: str,
    delta: str | None = None,
) -> None:
    delta_html = ""
    if delta:
        delta_html = (
            "<div style='font-size:0.82rem;font-weight:700;opacity:0.86;'>"
            f"{html.escape(delta)}"
            "</div>"
        )
    container.markdown(
        (
            "<div style='"
            f"background:{color};"
            "color:white;"
            "border-radius:8px;"
            "padding:14px 16px;"
            "min-height:94px;"
            "box-shadow:0 8px 22px rgba(15,23,42,0.12);"
            "display:flex;"
            "flex-direction:column;"
            "justify-content:space-between;"
            "'>"
            "<div style='font-size:0.78rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0;opacity:0.82;'>"
            f"{html.escape(label)}"
            "</div>"
            "<div style='font-size:1.6rem;font-weight:800;line-height:1.15;"
            "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
            f"{html.escape(str(value))}"
            "</div>"
            f"{delta_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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


def truncate_text(text: object, max_chars: int = CONTEXT_TEXT_LIMIT) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


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
                    color="articles",
                    color_continuous_scale=["#D9F3F7", "#0891B2", "#064E63"],
                ).update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title="Articles",
                    yaxis_title="Topic",
                    coloraxis_showscale=False,
                    showlegend=False,
                ).update_traces(
                    cliponaxis=False,
                    textposition="outside",
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


def _render_sentiment_analysis(
    df: pd.DataFrame,
    date_range: tuple[object, object] | None = None,
    synthetic_fill: bool = False,
) -> None:
    st.subheader("Sentiment Analysis")

    sentiment_df = _sentiment_distribution_dataframe(df)
    daily_sentiment_df = _daily_sentiment_percentage_dataframe(
        df,
        date_range=date_range,
        synthetic_fill=synthetic_fill,
    )

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
                    color_discrete_map=SENTIMENT_COLORS,
                ),
                use_container_width=True,
            )

    with evolution_col:
        if daily_sentiment_df.empty:
            st.info("Not enough dated articles to plot sentiment evolution.")
        else:
            st.plotly_chart(
                _sentiment_trend_figure(daily_sentiment_df),
                use_container_width=True,
            )


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


def _sentiment_trend_figure(daily_sentiment_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for sentiment, label in SENTIMENT_LABELS.items():
        sentiment_df = daily_sentiment_df[
            daily_sentiment_df["sentiment"] == sentiment
        ].sort_values("date")
        if sentiment_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sentiment_df["date"],
                y=sentiment_df["percentage"],
                name=label,
                mode="lines",
                line={
                    "color": SENTIMENT_COLORS[sentiment],
                    "width": 2,
                    "shape": "spline",
                    "smoothing": 0.7,
                },
                hovertemplate=f"{label} Sentiment<br><b>%{{y:.0f}}%</b><extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "Sentiment Trend Overview",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 16, "color": "#2F2F2F"},
        },
        height=320,
        margin={"l": 8, "r": 16, "t": 54, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.16,
            "yanchor": "top",
            "title": {"text": ""},
            "font": {"size": 13, "color": "#8A8A8A"},
        },
        yaxis={
            "title": {"text": "Sentiment Percentage"},
            "range": [0, 100],
            "ticksuffix": "%",
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.10)",
            "zeroline": False,
            "tickfont": {"color": "#8A8A8A"},
        },
        xaxis={
            "title": {"text": "Date"},
            "tickformat": "%d %b",
            "nticks": 8,
            "showgrid": False,
            "showline": False,
            "tickfont": {"color": "#8A8A8A"},
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikecolor": "rgba(0,0,0,0.18)",
            "spikethickness": 1,
        },
    )
    return fig


def _daily_sentiment_percentage_dataframe(
    df: pd.DataFrame,
    date_range: tuple[object, object] | None = None,
    synthetic_fill: bool = False,
) -> pd.DataFrame:
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(
        trend_df["date"], utc=True, errors="coerce")
    trend_df["sentiment"] = (
        trend_df["sentiment"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    trend_df = trend_df[
        trend_df["sentiment"].isin(["positive", "neutral", "negative"])
    ].dropna(subset=["date"])
    if trend_df.empty:
        return pd.DataFrame(
            columns=["date", "sentiment", "articles", "percentage", "data_type"]
        )

    trend_df["date"] = trend_df["date"].dt.date
    counts = (
        trend_df.groupby(["date", "sentiment"], as_index=False)
        .agg(articles=("title", "count"))
        .sort_values("date")
    )
    all_daily_sentiments = pd.MultiIndex.from_product(
        [
            sorted(counts["date"].unique()),
            ["positive", "neutral", "negative"],
        ],
        names=["date", "sentiment"],
    ).to_frame(index=False)
    counts = all_daily_sentiments.merge(
        counts,
        on=["date", "sentiment"],
        how="left",
    )
    counts["articles"] = counts["articles"].fillna(0).astype(int)
    daily_totals = counts.groupby("date")["articles"].transform("sum")
    counts["percentage"] = counts["articles"] / daily_totals * 100
    counts["data_type"] = "actual"
    if synthetic_fill:
        counts = _fill_synthetic_sentiment_dates(counts, date_range)
    return _smooth_sentiment_percentages(counts)


def _fill_synthetic_sentiment_dates(
    counts: pd.DataFrame,
    date_range: tuple[object, object] | None,
) -> pd.DataFrame:
    if counts.empty:
        return counts

    start_date = counts["date"].min()
    end_date = counts["date"].max()
    if date_range is not None:
        start_date, end_date = date_range

    all_dates = pd.date_range(start_date, end_date, freq="D").date
    all_daily_sentiments = pd.MultiIndex.from_product(
        [all_dates, ["positive", "neutral", "negative"]],
        names=["date", "sentiment"],
    ).to_frame(index=False)
    filled = all_daily_sentiments.merge(
        counts,
        on=["date", "sentiment"],
        how="left",
    )
    filled["articles"] = filled["articles"].fillna(0).astype(int)
    filled["data_type"] = filled["data_type"].fillna("synthetic")
    filled["percentage"] = (
        filled.groupby("sentiment")["percentage"]
        .transform(lambda values: values.interpolate().bfill().ffill())
        .fillna(0.0)
    )
    return _add_synthetic_sentiment_variation(filled)


def _add_synthetic_sentiment_variation(filled: pd.DataFrame) -> pd.DataFrame:
    synthetic_mask = filled["data_type"].eq("synthetic")
    if not synthetic_mask.any():
        return filled

    varied = filled.copy()
    phases = {
        "positive": 0.0,
        "neutral": 2.2,
        "negative": 4.4,
    }

    def adjusted_percentage(row: pd.Series) -> float:
        day_number = pd.Timestamp(row["date"]).toordinal()
        phase = phases.get(str(row["sentiment"]), 0.0)
        trend_noise = math.sin(day_number * 0.37 + phase) * 4.0
        small_noise = math.sin(day_number * 1.13 + phase * 1.7) * 1.6
        return max(1.0, float(row["percentage"]) + trend_noise + small_noise)

    varied.loc[synthetic_mask, "percentage"] = varied.loc[
        synthetic_mask
    ].apply(adjusted_percentage, axis=1)
    synthetic_dates = varied.loc[synthetic_mask, "date"].unique()
    date_mask = varied["date"].isin(synthetic_dates)
    daily_totals = varied.loc[date_mask].groupby("date")["percentage"].transform("sum")
    varied.loc[date_mask, "percentage"] = (
        varied.loc[date_mask, "percentage"] / daily_totals * 100
    )
    return varied


def _smooth_sentiment_percentages(counts: pd.DataFrame) -> pd.DataFrame:
    smoothed = counts.sort_values(["sentiment", "date"]).copy()
    smoothed["percentage"] = smoothed.groupby("sentiment")["percentage"].transform(
        lambda values: values.rolling(window=7, min_periods=1, center=True).mean()
    )
    daily_totals = smoothed.groupby("date")["percentage"].transform("sum")
    smoothed["percentage"] = smoothed["percentage"] / daily_totals * 100
    return smoothed.sort_values(["date", "sentiment"])


def _render_importance_analysis(df: pd.DataFrame) -> None:
    st.subheader("Article Importance")

    top_col, distribution_col = st.columns((1.4, 1))
    with top_col:
        st.markdown("**Importance moyenne dans le temps par compagnie**")
        company_importance_df = _company_importance_trend_dataframe(df)
        if company_importance_df.empty:
            st.info("Not enough dated company importance data yet.")
        else:
            importance_min = float(company_importance_df["avg_importance"].min())
            importance_max = float(company_importance_df["avg_importance"].max())
            importance_padding = max((importance_max - importance_min) * 0.12, 3.0)
            importance_range = [
                max(0.0, importance_min - importance_padding),
                min(100.0, importance_max + importance_padding),
            ]
            st.plotly_chart(
                px.line(
                    company_importance_df,
                    x="date",
                    y="avg_importance",
                    color="company",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data={
                        "company": False,
                        "articles": True,
                        "avg_importance": ":.1f",
                    },
                ).update_traces(
                    line={"width": 3, "shape": "spline"},
                ).update_layout(
                    xaxis_title="Date",
                    yaxis_title="Average importance",
                    yaxis={"range": importance_range},
                    legend_title_text="Company",
                ),
                use_container_width=True,
            )

    with distribution_col:
        st.markdown("**Importance score distribution**")
        distribution_df = _company_importance_distribution_dataframe(df)
        if distribution_df.empty:
            st.info("No company importance distribution available yet.")
        else:
            st.plotly_chart(
                px.bar(
                    distribution_df,
                    x="bucket",
                    y="articles",
                    color="company",
                    category_orders={
                        "bucket": ["0-20", "20-40", "40-60", "60-80", "80-100"]
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data={"company": True, "articles": True},
                ).update_layout(
                    barmode="stack",
                    xaxis_title="Importance score",
                    yaxis_title="Articles",
                    legend_title_text="Company",
                ),
                use_container_width=True,
            )

    st.markdown("**Top 5 articles importants**")
    _render_top_importance_article_cards(df)

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


def _render_top_importance_article_cards(df: pd.DataFrame) -> None:
    articles = _top_importance_articles(df)
    if not articles:
        st.info("No important articles available yet.")
        return

    cards = []
    for article in articles:
        title = html.escape(article["title"])
        source = html.escape(article["source"])
        date = html.escape(article["date"])
        score = html.escape(article["score"])
        url = html.escape(article["url"], quote=True)
        card_content = (
            "<div style='height:100%;border:1px solid rgba(8,145,178,0.18);"
            "border-radius:8px;padding:14px 14px 12px;background:white;"
            "box-shadow:0 8px 22px rgba(15,23,42,0.08);'>"
            "<div style='display:flex;justify-content:space-between;gap:10px;"
            "align-items:center;margin-bottom:10px;'>"
            "<span style='font-size:0.76rem;font-weight:800;color:#0891B2;"
            "text-transform:uppercase;letter-spacing:0;'>Importance</span>"
            "<span style='background:#0891B2;color:white;border-radius:999px;"
            "padding:3px 9px;font-size:0.78rem;font-weight:800;'>"
            f"{score}</span>"
            "</div>"
            "<div style='font-size:1rem;font-weight:800;line-height:1.25;"
            "color:#1F2937;margin-bottom:12px;'>"
            f"{title}</div>"
            "<div style='font-size:0.82rem;color:#6B7280;line-height:1.35;'>"
            f"{source}<br>{date}</div>"
            "</div>"
        )
        if url:
            cards.append(
                "<a href='"
                f"{url}"
                "' target='_blank' rel='noopener noreferrer' "
                "style='text-decoration:none;color:inherit;'>"
                f"{card_content}</a>"
            )
        else:
            cards.append(card_content)

    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));"
        "gap:12px;margin:8px 0 18px;'>"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _top_importance_articles(df: pd.DataFrame, limit: int = 5) -> list[dict[str, str]]:
    if df.empty:
        return []

    top_df = df.sort_values("importance_score", ascending=False).head(limit)
    articles: list[dict[str, str]] = []
    for _, article in top_df.iterrows():
        raw_date = pd.to_datetime(article.get("date"), utc=True, errors="coerce")
        date = raw_date.strftime("%Y-%m-%d") if not pd.isna(raw_date) else ""
        score = float(article.get("importance_score", 0.0) or 0.0)
        articles.append(
            {
                "title": _shorten_text(
                    str(article.get("title", "")).strip() or "Untitled article",
                    95,
                ),
                "source": str(article.get("source", "")).strip() or "Unknown source",
                "date": date,
                "score": f"{score:.1f}",
                "url": str(article.get("url", "")).strip(),
            }
        )
    return articles


def _company_importance_trend_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    mentions = _company_mentions_dataframe(df)
    if mentions.empty:
        return pd.DataFrame(
            columns=["date", "company", "avg_importance", "articles"]
        )

    mentions["date"] = pd.to_datetime(
        mentions["date"], utc=True, errors="coerce")
    mentions["importance_score"] = pd.to_numeric(
        mentions["importance_score"], errors="coerce")
    mentions = mentions.dropna(subset=["date", "importance_score"])
    if mentions.empty:
        return pd.DataFrame(
            columns=["date", "company", "avg_importance", "articles"]
        )

    top_companies = (
        mentions.groupby("company")["title"]
        .count()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    mentions = mentions[mentions["company"].isin(top_companies)].copy()
    mentions["date"] = mentions["date"].dt.date
    return (
        mentions.groupby(["date", "company"], as_index=False)
        .agg(
            avg_importance=("importance_score", "mean"),
            articles=("title", "count"),
        )
        .sort_values("date")
    )


def _company_importance_distribution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    buckets = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    mentions = _company_mentions_dataframe(df)
    if mentions.empty:
        return pd.DataFrame(columns=["bucket", "company", "articles"])

    mentions["importance_score"] = pd.to_numeric(
        mentions["importance_score"], errors="coerce")
    mentions = mentions.dropna(subset=["importance_score"])
    if mentions.empty:
        return pd.DataFrame(columns=["bucket", "company", "articles"])

    top_companies = (
        mentions.groupby("company")["title"]
        .count()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    mentions = mentions[mentions["company"].isin(top_companies)].copy()
    mentions["bucket"] = pd.cut(
        mentions["importance_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=buckets,
        include_lowest=True,
        right=True,
    ).astype(str)
    return (
        mentions.groupby(["bucket", "company"], as_index=False)
        .agg(articles=("title", "count"))
        .sort_values(["bucket", "articles"], ascending=[True, False])
    )


def _render_sources_companies(df: pd.DataFrame) -> None:
    st.subheader("Sources & Companies")

    source_col, company_col = st.columns(2)
    source_df = _source_strategy_dataframe(df)
    with source_col:
        st.markdown("**Most active sources**")
        if source_df.empty:
            st.info("No source data available yet.")
        else:
            source_donut_df = _donut_breakdown_dataframe(
                source_df, "Source", "articles")
            st.plotly_chart(
                px.pie(
                    source_donut_df,
                    names="Source",
                    values="articles",
                    hole=0.58,
                    color_discrete_sequence=DONUT_COLORS,
                ).update_layout(
                    showlegend=False,
                    margin={"l": 16, "r": 16, "t": 8, "b": 8},
                    height=360,
                ).update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    marker={"line": {"color": "white", "width": 2}},
                    hovertemplate="%{label}<br>Articles: %{value}<br>%{percent}<extra></extra>",
                ),
                use_container_width=True,
            )

    company_df = _company_strategy_dataframe(df)
    with company_col:
        st.markdown("**Most mentioned companies**")
        if company_df.empty:
            st.info("No tracked company mentions detected yet.")
        else:
            company_donut_df = _donut_breakdown_dataframe(
                company_df, "Company", "Articles")
            st.plotly_chart(
                px.pie(
                    company_donut_df,
                    names="Company",
                    values="Articles",
                    hole=0.58,
                    color_discrete_sequence=DONUT_COLORS,
                ).update_layout(
                    showlegend=False,
                    margin={"l": 16, "r": 16, "t": 8, "b": 8},
                    height=360,
                ).update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    marker={"line": {"color": "white", "width": 2}},
                    hovertemplate="%{label}<br>Articles: %{value}<br>%{percent}<extra></extra>",
                ),
                use_container_width=True,
            )


def _source_strategy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["Source", "articles", "Avg sentiment", "Avg importance"]
        )

    source_input = df
    if "sentiment_score" not in source_input.columns:
        source_input = _add_confidence_scores(source_input)

    source_df = (
        source_input.groupby("source", as_index=False)
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


def _donut_breakdown_dataframe(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    limit: int = 5,
) -> pd.DataFrame:
    chart_df = df[[label_column, value_column]].copy()
    chart_df[value_column] = pd.to_numeric(
        chart_df[value_column], errors="coerce").fillna(0)
    chart_df = chart_df[chart_df[value_column] > 0].sort_values(
        value_column, ascending=False
    )
    if len(chart_df) <= limit:
        return chart_df

    top_df = chart_df.head(limit).copy()
    other_value = chart_df.iloc[limit:][value_column].sum()
    if other_value <= 0:
        return top_df

    other_df = pd.DataFrame(
        [{label_column: "Other", value_column: other_value}]
    )
    return pd.concat([top_df, other_df], ignore_index=True)


def _company_mentions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, article in df.iterrows():
        for company in _detect_companies(article):
            rows.append(
                {
                    "company": company,
                    "title": str(article.get("title", "")),
                    "date": article.get("date"),
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


def build_dashboard_context(filtered_df: pd.DataFrame) -> dict[str, object]:
    return build_compact_dashboard_context(filtered_df)


def build_compact_dashboard_context(filtered_df: pd.DataFrame) -> dict[str, object]:
    if filtered_df.empty:
        return {
            "total_articles": 0,
            "date_range": {"start": None, "end": None},
            "sentiment_distribution": [],
            "sentiment_by_day": [],
            "average_importance_score": 0.0,
            "top_important_articles": [],
            "top_sources": [],
            "top_companies": [],
            "top_topics": [],
            "negative_high_importance_articles": [],
            "weak_signals": [],
        }

    date_series = pd.to_datetime(
        filtered_df.get("date", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    ).dropna()
    date_range = {
        "start": date_series.min().date().isoformat() if not date_series.empty else None,
        "end": date_series.max().date().isoformat() if not date_series.empty else None,
    }
    sentiment_by_day = _last_sentiment_days(
        _daily_sentiment_percentage_dataframe(filtered_df),
        CONTEXT_SENTIMENT_DAYS,
    )
    sentiment_distribution = _sentiment_distribution_dataframe(filtered_df)
    return {
        "total_articles": int(len(filtered_df)),
        "date_range": date_range,
        "sentiment_distribution": _compact_records(sentiment_distribution),
        "average_importance_score": _average_importance_score(filtered_df),
        "dominant_sentiment": _dominant_sentiment_from_distribution(sentiment_distribution),
        "sentiment_by_day": _compact_records(sentiment_by_day),
        "top_important_articles": _compact_article_records(filtered_df, limit=5),
        "negative_high_importance_articles": _negative_high_importance_articles(
            filtered_df, limit=5
        ),
        "top_sources": _compact_records(_source_strategy_dataframe(filtered_df).head(10)),
        "top_companies": _compact_records(_company_strategy_dataframe(filtered_df).head(10)),
        "top_topics": _compact_records(_topic_counts_dataframe(filtered_df).head(10)),
        "weak_signals": _compact_records(_weak_signals_dataframe(filtered_df)),
    }


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    return _compact_records(df, text_limit=None)


def _compact_records(
    df: pd.DataFrame,
    *,
    text_limit: int | None = CONTEXT_TEXT_LIMIT,
) -> list[dict[str, object]]:
    if df.empty:
        return []
    clean_df = df.copy()
    for column in clean_df.columns:
        if pd.api.types.is_datetime64_any_dtype(clean_df[column]):
            clean_df[column] = clean_df[column].astype(str)
        if text_limit is not None and clean_df[column].dtype == object:
            clean_df[column] = clean_df[column].apply(
                lambda value: truncate_text(value, text_limit)
            )
    return clean_df.to_dict("records")


def _last_sentiment_days(sentiment_df: pd.DataFrame, days: int) -> pd.DataFrame:
    if sentiment_df.empty:
        return sentiment_df
    scoped_df = sentiment_df.copy()
    scoped_df["date"] = pd.to_datetime(scoped_df["date"], errors="coerce")
    scoped_df = scoped_df.dropna(subset=["date"]).sort_values("date")
    last_dates = scoped_df["date"].dt.date.drop_duplicates().tail(days)
    return scoped_df[scoped_df["date"].dt.date.isin(set(last_dates))]


def _dominant_sentiment_from_distribution(sentiment_df: pd.DataFrame) -> str:
    if sentiment_df.empty:
        return "unknown"
    top_row = sentiment_df.sort_values("articles", ascending=False).iloc[0]
    return str(top_row.get("sentiment", "unknown"))


def _average_importance_score(df: pd.DataFrame) -> float:
    if df.empty or "importance_score" not in df.columns:
        return 0.0
    scores = pd.to_numeric(df["importance_score"], errors="coerce").dropna()
    if scores.empty:
        return 0.0
    return round(float(scores.mean()), 1)


def _compact_article_records(
    df: pd.DataFrame,
    *,
    limit: int,
    include_url: bool = False,
) -> list[dict[str, object]]:
    if df.empty:
        return []
    top_df = df.sort_values("importance_score", ascending=False).head(limit)
    articles: list[dict[str, object]] = []
    for _, article in top_df.iterrows():
        raw_date = pd.to_datetime(article.get("date"), utc=True, errors="coerce")
        record: dict[str, object] = {
            "title": truncate_text(article.get("title")),
            "source": truncate_text(article.get("source")),
            "date": raw_date.strftime("%Y-%m-%d") if not pd.isna(raw_date) else "",
            "sentiment": truncate_text(article.get("sentiment"), 40),
            "importance_score": round(
                float(article.get("importance_score", 0.0) or 0.0), 1
            ),
            "summary": truncate_text(
                article.get("summary") or article.get("description")
            ),
            "keywords": normalize_keywords(article.get("keywords"))[:8],
            "companies": _detect_companies(article)[:5],
        }
        extracted_entities = article.get("extracted_entities")
        if isinstance(extracted_entities, list):
            record["extracted_entities"] = [
                truncate_text(entity, 80) for entity in extracted_entities[:5]
            ]
        if include_url:
            record["url"] = truncate_text(article.get("url"), 300)
        articles.append(record)
    return articles


def _negative_high_importance_articles(
    df: pd.DataFrame,
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    if df.empty:
        return []
    negative_df = df[
        df["sentiment"].fillna("").astype(str).str.lower().eq("negative")
        & (pd.to_numeric(df["importance_score"], errors="coerce").fillna(0.0) >= 70)
    ].copy()
    return _compact_article_records(negative_df, limit=limit)


def _filter_assistant_dataframe_by_time(
    df: pd.DataFrame,
    prompt: str,
) -> tuple[pd.DataFrame, str | None]:
    window = _assistant_temporal_window(prompt)
    if window is None or df.empty:
        return df, None

    start_date, end_date, label = window
    scoped_df = df.copy()
    scoped_df["date_dt"] = pd.to_datetime(scoped_df["date"], utc=True, errors="coerce")
    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    scoped_df = scoped_df[
        (scoped_df["date_dt"] >= start_dt) & (scoped_df["date_dt"] < end_dt)
    ].drop(columns=["date_dt"])
    return scoped_df, label


def _assistant_temporal_window(
    prompt: str,
    *,
    now: datetime | None = None,
) -> tuple[object, object, str] | None:
    normalized = _normalize_prompt(prompt)
    now = now or datetime.now(timezone.utc)
    today = now.date()

    if "today" in normalized or "aujourd'hui" in normalized or "aujourdhui" in normalized:
        return today, today, "today"
    if "yesterday" in normalized or "hier" in normalized:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday, "yesterday"
    if "last 7 days" in normalized or "ces derniers jours" in normalized:
        return today - timedelta(days=6), today, "last 7 days"
    if "this week" in normalized:
        start = today - timedelta(days=today.weekday())
        return start, today, "this week"
    if "last week" in normalized or "la semaine derniere" in normalized:
        this_week_start = today - timedelta(days=today.weekday())
        last_week_end = this_week_start - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        return last_week_start, last_week_end, "last week"
    return None


def _normalize_prompt(prompt: str) -> str:
    return (
        prompt.lower()
        .replace("’", "'")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
    )


def _question_for_follow_up(intent: str, prompt: str, language: str = "en") -> str:
    base_questions_by_language = {
        "en": {
            "trends": "What are the strongest AI Pulse trends?",
            "sentiment": "What changed in AI Pulse sentiment over time?",
            "companies": "Which companies are most mentioned in AI Pulse?",
            "sources": "Which sources are most active in AI Pulse?",
            "importance": "Which AI Pulse articles are most important?",
            "weak_signals": "What weak signals should I watch in AI Pulse?",
            "general_summary": "What happened in AI Pulse?",
        },
        "fr": {
            "trends": "Quelles sont les tendances AI Pulse les plus fortes ?",
            "sentiment": "Comment le sentiment AI Pulse a-t-il évolué ?",
            "companies": "Quelles companies sont les plus mentionnées dans AI Pulse ?",
            "sources": "Quelles sources sont les plus actives dans AI Pulse ?",
            "importance": "Quels articles AI Pulse sont les plus importants ?",
            "weak_signals": "Quels signaux faibles faut-il surveiller dans AI Pulse ?",
            "general_summary": "Que s’est-il passé dans AI Pulse ?",
        },
    }
    base_questions = base_questions_by_language.get(language, base_questions_by_language["en"])
    return f"{base_questions.get(intent, base_questions['general_summary'])} {prompt}"


def _set_browser_title(title: str) -> None:
    components.html(
        f"<script>window.parent.document.title = {json.dumps(title)};</script>",
        height=0,
        width=0,
    )


def _render_ai_assistant(df: pd.DataFrame) -> None:
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "assistant_answer_cache" not in st.session_state:
        st.session_state.assistant_answer_cache = {}
    if "assistant_language" not in st.session_state:
        st.session_state.assistant_language = None

    current_language = str(st.session_state.get("assistant_language") or "en")
    _set_browser_title(
        "Assistant AI Pulse" if current_language == "fr" else "AI Pulse Assistant"
    )
    st.subheader("Assistant AI Pulse" if current_language == "fr" else "AI Pulse Assistant")
    st.caption(
        (
            "Pose une question sur les tendances du dashboard, companies, sources, "
            "sentiment, articles importants ou signaux faibles."
        )
        if current_language == "fr"
        else (
            "Ask about dashboard trends, companies, sources, sentiment, "
            "important articles, or weak signals."
        )
    )

    if st.button("Clear chat", key="clear-agent-chat"):
        st.session_state.agent_messages = []
        st.session_state.assistant_language = None
        st.session_state["assistant_last_intent"] = None

    selected_prompt = None
    for index, message in enumerate(st.session_state.agent_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_evidence(message.get("evidence", []))
            if message.get("suggestions"):
                message_language = str(
                    message.get("language") or st.session_state.get("assistant_language") or "en"
                )
                selected_prompt = (
                    _render_assistant_suggestions(
                        message["suggestions"],
                        f"assistant-message-{index}-suggestion",
                        language=message_language,
                    )
                    or selected_prompt
                )

    typed_prompt = st.chat_input(
        (
            "Pose une question sur les tendances IA, companies, RAG, agents..."
            if current_language == "fr"
            else "Ask about AI trends, companies, RAG, agents..."
        ),
        key="ai-agent-chat-input",
    )
    prompt = selected_prompt or typed_prompt
    if not prompt:
        return

    requested_language = detect_language_change_request(prompt)
    if requested_language:
        st.session_state.assistant_language = requested_language
    elif not st.session_state.get("assistant_language"):
        st.session_state.assistant_language = detect_assistant_language(prompt)
    assistant_language = str(st.session_state.get("assistant_language") or "en")

    detected_intent = detect_dashboard_intent(prompt)
    last_intent = st.session_state.get("assistant_last_intent")
    if detected_intent == "follow_up":
        effective_intent = str(last_intent or "general_summary")
        effective_question = _question_for_follow_up(
            effective_intent,
            prompt,
            language=assistant_language,
        )
    else:
        effective_intent = detected_intent
        effective_question = prompt

    if detected_intent != "follow_up" and is_conversation_prompt(prompt):
        search_results = []
        answer = answer_conversation(prompt, language=assistant_language)
    else:
        scoped_df, temporal_label = _filter_assistant_dataframe_by_time(
            df,
            effective_question,
        )
        dashboard_context = build_compact_dashboard_context(scoped_df)
        if temporal_label:
            dashboard_context["temporal_filter"] = temporal_label
        search_results = search_articles(
            scoped_df.to_dict("records"),
            effective_question,
            limit=3,
        )
        cache_key = _assistant_answer_cache_key(
            effective_question,
            effective_intent,
            dashboard_context,
            assistant_language,
        )
        cached_answer = st.session_state.assistant_answer_cache.get(cache_key)
        if cached_answer:
            answer = cached_answer
        else:
            answer = answer_dashboard_question(
                effective_question,
                dashboard_context,
                search_results,
                intent=effective_intent,
                language=assistant_language,
            )
            st.session_state.assistant_answer_cache[cache_key] = answer
        st.session_state["assistant_last_intent"] = effective_intent
    suggestions = _assistant_suggestions_for_prompt(
        effective_question,
        language=assistant_language,
    )

    user_message = {"role": "user", "content": prompt}
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "evidence": search_results,
        "suggestions": suggestions,
        "language": assistant_language,
    }
    st.session_state.agent_messages.extend([user_message, assistant_message])

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        _render_evidence(search_results)
        if suggestions:
            _render_assistant_suggestions(
                suggestions,
                f"assistant-live-suggestion-{len(st.session_state.agent_messages)}",
                language=assistant_language,
            )


def _assistant_answer_cache_key(
    question: str,
    intent: str,
    dashboard_context: dict[str, object],
    language: str,
) -> str:
    payload = json.dumps(
        {
            "question": question,
            "intent": intent,
            "language": language,
            "dashboard_context": dashboard_context,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _render_assistant_suggestions(
    suggestions: list[str],
    key_prefix: str,
    language: str = "en",
) -> str | None:
    label = "Questions dashboard suggérées" if language == "fr" else "Suggested dashboard questions"
    st.markdown(f"**{label}**")
    selected_prompt = None
    columns = st.columns(2)
    for index, question in enumerate(suggestions):
        column = columns[index % len(columns)]
        with column:
            if st.button(question, key=f"{key_prefix}-{index}", use_container_width=True):
                selected_prompt = question
    return selected_prompt


def _assistant_suggestions_for_prompt(prompt: str, language: str = "en") -> list[str]:
    normalized = prompt.lower()
    suggestion_groups_by_language = {
        "en": [
            (
                ("sentiment", "positive", "negative", "neutral", "positif", "negatif"),
                [
                    "What explains this AI Pulse sentiment trend?",
                    "Which AI Pulse sources drive this sentiment?",
                    "Which companies have the strongest sentiment shift?",
                ],
            ),
            (
                ("company", "companies", "entreprise", "entreprises", "openai", "google", "microsoft", "anthropic"),
                [
                    "Which AI Pulse companies are gaining visibility?",
                    "Compare company sentiment in AI Pulse.",
                    "Which important articles mention these companies?",
                ],
            ),
            (
                ("source", "sources", "media", "publisher"),
                [
                    "Which AI Pulse sources publish the most articles?",
                    "Which sources cover the highest-importance topics?",
                    "How does sentiment differ by source?",
                ],
            ),
            (
                ("importance", "important", "article", "articles"),
                [
                    "Which AI Pulse articles are most important?",
                    "What topics appear in high-importance articles?",
                    "Which companies dominate important articles?",
                ],
            ),
            (
                ("signal", "signals", "weak", "signaux", "faibles"),
                [
                    "What weak signals should I watch in AI Pulse?",
                    "Which low-volume topics look important?",
                    "Which companies appear in weak signals?",
                ],
            ),
            (
                ("trend", "trends", "tendance", "tendances", "topic", "topics", "rag", "agent", "agents"),
                [
                    "Which AI Pulse topics are trending fastest?",
                    "How are RAG and AI agents evolving?",
                    "Which companies are linked to these trends?",
                ],
            ),
        ],
        "fr": [
            (
                ("sentiment", "positive", "negative", "neutral", "positif", "negatif"),
                [
                    "Qu’est-ce qui explique cette tendance de sentiment AI Pulse ?",
                    "Quelles sources AI Pulse influencent ce sentiment ?",
                    "Quelles companies ont le plus fort changement de sentiment ?",
                ],
            ),
            (
                ("company", "companies", "entreprise", "entreprises", "openai", "google", "microsoft", "anthropic"),
                [
                    "Quelles companies gagnent en visibilité dans AI Pulse ?",
                    "Compare le sentiment des companies dans AI Pulse.",
                    "Quels articles importants mentionnent ces companies ?",
                ],
            ),
            (
                ("source", "sources", "media", "publisher"),
                [
                    "Quelles sources AI Pulse publient le plus d’articles ?",
                    "Quelles sources couvrent les sujets les plus importants ?",
                    "Comment le sentiment varie-t-il par source ?",
                ],
            ),
            (
                ("importance", "important", "article", "articles"),
                [
                    "Quels articles AI Pulse sont les plus importants ?",
                    "Quels topics apparaissent dans les articles à forte importance ?",
                    "Quelles companies dominent les articles importants ?",
                ],
            ),
            (
                ("signal", "signals", "weak", "signaux", "faibles"),
                [
                    "Quels signaux faibles dois-je surveiller dans AI Pulse ?",
                    "Quels topics peu visibles semblent importants ?",
                    "Quelles companies apparaissent dans les signaux faibles ?",
                ],
            ),
            (
                ("trend", "trends", "tendance", "tendances", "topic", "topics", "rag", "agent", "agents"),
                [
                    "Quels topics AI Pulse progressent le plus vite ?",
                    "Comment évoluent RAG et les agents IA ?",
                    "Quelles companies sont liées à ces tendances ?",
                ],
            ),
        ],
    }
    suggestion_groups = suggestion_groups_by_language.get(
        language,
        suggestion_groups_by_language["en"],
    )

    for keywords, suggestions in suggestion_groups:
        if any(keyword in normalized for keyword in keywords):
            return suggestions[:3]
    return ASSISTANT_DEFAULT_SUGGESTIONS_BY_LANGUAGE.get(
        language,
        ASSISTANT_DEFAULT_SUGGESTIONS_BY_LANGUAGE["en"],
    )[:3]


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
    styled_explorer = explorer_df.style.apply(_style_article_explorer_row, axis=1)
    table_height = min(760, max(260, 78 + len(explorer_df) * 42))
    st.dataframe(
        styled_explorer,
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Source": st.column_config.TextColumn("Source", width="small"),
            "Topic": st.column_config.TextColumn("Topic", width="small"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Sentiment score": st.column_config.NumberColumn(
                "Sentiment score",
                format="%.2f",
                width="small",
            ),
            "Importance score": st.column_config.ProgressColumn(
                "Importance score",
                min_value=0,
                max_value=100,
                format="%.1f",
                width="medium",
            ),
            "URL": st.column_config.LinkColumn(
                "Article",
                display_text="Open",
                width="small",
            ),
            "Summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )


def _style_article_explorer_row(row: pd.Series) -> list[str]:
    sentiment = str(row.get("Sentiment", "")).lower()
    sentiment_style = {
        "positive": "background-color:#E9F9EF;color:#166534;font-weight:800;",
        "neutral": "background-color:#FFF4E6;color:#9A3412;font-weight:800;",
        "negative": "background-color:#FEECEC;color:#991B1B;font-weight:800;",
    }.get(sentiment, "background-color:#F8FAFC;color:#475569;font-weight:700;")

    styles: list[str] = []
    for column in row.index:
        if column == "Sentiment":
            styles.append(sentiment_style)
        elif column == "Title":
            styles.append("font-weight:800;color:#111827;")
        elif column == "Source":
            styles.append("color:#0891B2;font-weight:700;")
        elif column == "Topic":
            styles.append("color:#7C3AED;font-weight:700;")
        elif column == "Summary":
            styles.append("color:#475569;")
        else:
            styles.append("")
    return styles


def _render_dashboard_header(df: pd.DataFrame) -> None:
    date_series = pd.to_datetime(
        df["date"], utc=True, errors="coerce").dropna()
    caption_col, update_col = st.columns([1, 0.12])
    if date_series.empty:
        with caption_col:
            st.caption("Last update: unknown")
        with update_col:
            if st.button("Update", key="dashboard-update-button", type="primary"):
                _run_dashboard_update(None)
        _render_dashboard_update_message()
        return
    last_update_dt = date_series.max().to_pydatetime()
    last_update = last_update_dt.strftime("%Y-%m-%d %H:%M UTC")
    with caption_col:
        st.caption(f"Last update: {last_update}")
    with update_col:
        if st.button("Update", key="dashboard-update-button", type="primary"):
            _run_dashboard_update(last_update_dt)
    _render_dashboard_update_message()


def _render_dashboard_update_message() -> None:
    message = st.session_state.pop("dashboard_update_message", None)
    if not message:
        return

    message_type, message_text = message
    if message_type == "success":
        st.success(message_text)
    else:
        st.info(message_text)


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


def _quick_date_range(
    selected_filter: str,
    max_date: object,
) -> tuple[object, object] | None:
    lookback = _date_filter_options().get(selected_filter)
    if lookback is None:
        return None
    return max_date - lookback, max_date


def _render_dashboard_view(df: pd.DataFrame) -> None:
    _render_dashboard_header(df)

    sentiments_available = sorted(df["sentiment"].dropna().unique().tolist())
    sources_available = sorted(df["source"].fillna(
        "").astype(str).unique().tolist())
    sources_available = [source for source in sources_available if source]
    max_importance = float(df["importance_score"].max()
                           ) if not df.empty else 0.0
    min_date, max_date = _date_bounds(df)
    visual_min_date = min_date - timedelta(days=365)

    col_date, col_sentiment, col_source, col_importance = st.columns(
        (1.2, 1, 1, 1))
    selected_date_filter = col_date.selectbox(
        "Date filter",
        list(_date_filter_options()),
        index=1,
    )
    sentiment_filter = col_sentiment.selectbox(
        "Sentiment",
        ["All sentiments", "Select sentiments"],
    )
    source_filter = col_source.selectbox(
        "Source",
        ["All sources", "Select sources"],
    )
    min_importance = col_importance.slider(
        "Min importance",
        min_value=0.0,
        max_value=max(100.0, max_importance),
        value=0.0,
        step=5.0,
    )
    if selected_date_filter == "Range":
        selected_date_range = st.date_input(
            "Custom date range",
            value=(min_date, max_date),
            min_value=visual_min_date,
            max_value=max_date,
        )
        normalized_date_range = _normalize_date_range(selected_date_range)
    else:
        normalized_date_range = _quick_date_range(selected_date_filter, max_date)

    if source_filter == "Select sources":
        selected_sources = st.multiselect(
            "Selected sources",
            sources_available,
            default=[],
        )
    else:
        selected_sources = []

    if sentiment_filter == "Select sentiments":
        selected_sentiments = st.multiselect(
            "Selected sentiments",
            sentiments_available,
            default=[],
        )
    else:
        selected_sentiments = []

    synthetic_fill = st.checkbox(
        "Fill missing dates for visual charts",
        value=True,
    )

    filtered_df = _apply_filters(
        df,
        selected_sentiments,
        selected_sources,
        min_importance,
        normalized_date_range,
    )

    if filtered_df.empty:
        st.info("No articles match the selected filters.")
        return

    filtered_df = _add_confidence_scores(filtered_df)
    _render_kpi_cards(filtered_df)
    st.divider()
    _render_sentiment_analysis(
        filtered_df,
        date_range=normalized_date_range,
        synthetic_fill=synthetic_fill,
    )
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
    _set_browser_title(
        "AI Pulse Dashboard"
        if selected_section == "Dashboard"
        else "AI Pulse Assistant"
    )

    df = load_dataframe()

    if df.empty:
        if selected_section == "Assistant":
            _render_ai_assistant(pd.DataFrame())
        else:
            st.warning(
                "No documents found in Cosmos DB. Run the ingestion pipeline first.")
        return

    df = _prepare_dataframe(df)
    if df.empty:
        if selected_section == "Assistant":
            _render_ai_assistant(pd.DataFrame())
        else:
            st.warning("No AI-related documents found in Cosmos DB.")
        return

    if selected_section == "Dashboard":
        _render_dashboard_view(df)
    elif selected_section == "Assistant":
        _render_ai_assistant(df)


if __name__ == "__main__":
    render_dashboard()
