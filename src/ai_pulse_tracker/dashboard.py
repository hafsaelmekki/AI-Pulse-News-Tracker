from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from .agent import answer_question
from .config import load_settings
from .models import UpsertResult
from .pipeline import NewsAnalyzerPipeline
from .retrieval import search_articles
from .trends import count_keywords, format_keywords, normalize_keywords

st.set_page_config(page_title="AI Pulse Tracker", layout="wide")
PIPELINE_RESOURCE_KEY = "pipeline-v2"


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
        st.sidebar.error("Invalid ISO8601 datetime. Example: 2024-04-02T09:30:00Z")
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
    timeframe_label: str,
    sentiments: list[str],
    keywords: list[str],
    min_importance: float,
) -> pd.DataFrame:
    filtered = df.copy()
    filtered["date_dt"] = pd.to_datetime(filtered["date"], utc=True, errors="coerce")
    window = _time_filters().get(timeframe_label)
    if window is not None:
        threshold = datetime.now(timezone.utc) - window
        filtered = filtered[filtered["date_dt"] >= threshold]

    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]

    filtered = filtered[filtered["importance_score"] >= min_importance]

    if keywords:
        title_text = filtered["title"].fillna("").str.lower()
        desc_text = filtered.get("description")
        if desc_text is not None:
            title_text = title_text + " " + desc_text.fillna("").str.lower()
        keyword_text = filtered["keywords"].apply(lambda values: " ".join(values))
        search_text = title_text + " " + keyword_text
        mask = search_text.apply(lambda text: any(keyword in text for keyword in keywords))
        filtered = filtered[mask]

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
    prepared["keyword_text"] = prepared["keywords"].apply(format_keywords)
    return prepared


def _keyword_dataframe(df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    return pd.DataFrame(count_keywords(df.to_dict("records"), limit=limit))


def render_dashboard() -> None:
    st.title("AI Pulse - Azure Sentiment Monitor")
    st.caption("Latest French-language AI coverage scored via Azure AI Language")

    st.sidebar.header("Real-time Controls")
    query_override = st.sidebar.text_input(
        "Override NewsAPI query",
        placeholder="Generative AI",
        help="Leave empty to use the default configured query.",
    )
    since_text = st.sidebar.text_input(
        "Fetch articles published after (ISO8601)",
        placeholder="2024-04-01T08:00:00Z",
    )
    full_refresh = st.sidebar.checkbox(
        "Ignore incremental cursor", help="Refetch even if articles already ingested."
    )
    refresh_clicked = st.sidebar.button("Refresh data cache", key="refresh-cache")
    fetch_clicked = st.sidebar.button(
        "Ingest latest articles", type="primary", key="fetch-latest"
    )

    status_message: str | None = None
    since_override = _parse_since_text(since_text)

    if fetch_clicked:
        with st.spinner("Contacting NewsAPI + Azure AI..."):
            raw_result = get_pipeline(PIPELINE_RESOURCE_KEY).run(
                query=query_override or None,
                after=since_override,
                incremental=not full_refresh and since_override is None,
            )
        result = _ensure_upsert_result(raw_result)
        load_dataframe.clear()
        status_message = (
            f"Ingested {result.created} new / {result.updated} refreshed "
            f"at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}."
        )
    elif refresh_clicked:
        load_dataframe.clear()

    df = load_dataframe()
    if status_message:
        st.success(status_message)

    if df.empty:
        st.warning("No documents found in Cosmos DB. Run the ingestion pipeline first.")
        return

    df = _prepare_dataframe(df)

    st.subheader("Filters")
    sentiments_available = sorted(df["sentiment"].dropna().unique().tolist())
    time_labels = list(_time_filters().keys())
    max_importance = float(df["importance_score"].max()) if not df.empty else 0.0
    col_time, col_sentiment, col_importance, col_topics = st.columns((1, 1, 1, 1.2))
    timeframe_label = col_time.selectbox(
        "Time window", time_labels, index=len(time_labels) - 1
    )
    selected_sentiments = col_sentiment.multiselect(
        "Sentiment",
        sentiments_available,
        default=sentiments_available,
    )
    min_importance = col_importance.slider(
        "Min importance",
        min_value=0.0,
        max_value=max(100.0, max_importance),
        value=0.0,
        step=5.0,
    )
    topics_input = col_topics.text_input(
        "Topics (comma-separated keywords)",
        placeholder="openai, regulation, healthcare",
    )
    keywords = [word.strip().lower() for word in topics_input.split(",") if word.strip()]

    df = _apply_filters(
        df,
        timeframe_label,
        selected_sentiments,
        keywords,
        min_importance,
    )

    if df.empty:
        st.info("No articles match the selected filters.")
        return

    df["pos_score"] = df["confidence"].apply(lambda row: row["pos"])  # type: ignore[index]
    df["neu_score"] = df["confidence"].apply(lambda row: row["neu"])  # type: ignore[index]
    df["neg_score"] = df["confidence"].apply(lambda row: row["neg"])  # type: ignore[index]

    sentiment_counts = df["sentiment"].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed Articles", int(len(df)))
    dominant = sentiment_counts.idxmax().capitalize()
    col2.metric("Dominant Sentiment", dominant)
    col3.metric("Unique Sources", int(df["source"].nunique()))
    col4.metric("Avg Importance", f"{df['importance_score'].mean():.1f}")

    st.divider()
    st.subheader("AI Trend Assistant")
    search_query = st.text_input(
        "Ask a question about stored articles",
        placeholder="What are the AI agent trends this week?",
    )
    if search_query.strip():
        search_results = search_articles(
            df.to_dict("records"),
            search_query,
            limit=5,
        )
        st.markdown(answer_question(search_query, search_results))
        if search_results:
            st.caption("Retrieved article evidence")
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

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(
            px.pie(
                df,
                names="sentiment",
                color="sentiment",
                color_discrete_map={
                    "positive": "#00CC96",
                    "neutral": "#636EFA",
                    "negative": "#EF553B",
                },
            ),
            use_container_width=True,
        )
    with c2:
        st.subheader("Average Confidence")
        avg_pos = df["pos_score"].mean()
        st.progress(avg_pos, text=f"Optimism index: {avg_pos:.0%}")
        st.caption("Scores reported by Azure AI Language")

    st.divider()
    trend_col, source_col = st.columns(2)
    with trend_col:
        st.subheader("Top Keywords")
        keyword_df = _keyword_dataframe(df)
        if keyword_df.empty:
            st.info("No keywords available yet. Ingest new articles to enrich the dataset.")
        else:
            st.plotly_chart(
                px.bar(
                    keyword_df,
                    x="count",
                    y="keyword",
                    orientation="h",
                    text="count",
                ).update_layout(yaxis={"categoryorder": "total ascending"}),
                use_container_width=True,
            )
    with source_col:
        st.subheader("Source Coverage")
        source_df = (
            df.groupby("source", as_index=False)
            .agg(articles=("title", "count"), avg_importance=("importance_score", "mean"))
            .sort_values("articles", ascending=False)
            .head(12)
        )
        st.plotly_chart(
            px.bar(
                source_df,
                x="articles",
                y="source",
                orientation="h",
                color="avg_importance",
                color_continuous_scale="Blues",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            use_container_width=True,
        )

    st.subheader("High Importance Articles")
    st.dataframe(
        df.sort_values("importance_score", ascending=False)[
            [
                "date",
                "source",
                "title",
                "summary",
                "sentiment",
                "importance_score",
                "keyword_text",
                "url",
            ]
        ].head(20),
        use_container_width=True,
    )

    st.subheader("Latest Articles")
    st.dataframe(
        df[
            [
                "date",
                "source",
                "title",
                "summary",
                "sentiment",
                "importance_score",
                "keyword_text",
                "url",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    render_dashboard()
