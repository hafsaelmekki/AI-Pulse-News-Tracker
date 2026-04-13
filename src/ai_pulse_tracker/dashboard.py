from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import load_settings
from .models import UpsertResult
from .pipeline import NewsAnalyzerPipeline

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

    if "source_name" in df.columns:
        df["source"] = df["source_name"].fillna(df["source"])

    df["pos_score"] = df["confidence"].apply(lambda row: row["pos"])  # type: ignore[index]
    df["neu_score"] = df["confidence"].apply(lambda row: row["neu"])  # type: ignore[index]
    df["neg_score"] = df["confidence"].apply(lambda row: row["neg"])  # type: ignore[index]

    sentiment_counts = df["sentiment"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed Articles", int(len(df)))
    dominant = sentiment_counts.idxmax().capitalize()
    col2.metric("Dominant Sentiment", dominant)
    col3.metric("Unique Sources", int(df["source"].nunique()))

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

    st.subheader("Latest Articles")
    st.dataframe(
        df[["date", "source", "title", "sentiment", "url"]],
        use_container_width=True,
    )


if __name__ == "__main__":
    render_dashboard()
