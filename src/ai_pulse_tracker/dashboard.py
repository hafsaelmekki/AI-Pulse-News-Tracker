from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import load_settings
from .pipeline import NewsAnalyzerPipeline

st.set_page_config(page_title="AI Pulse Tracker", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> NewsAnalyzerPipeline:
    settings = load_settings()
    return NewsAnalyzerPipeline(settings)


@st.cache_data(ttl=600, show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    rows = get_pipeline().load_dashboard_rows()
    return pd.DataFrame(rows)


def render_dashboard() -> None:
    st.title("AI Pulse - Azure Sentiment Monitor")
    st.caption("Latest French-language AI coverage scored via Azure AI Language")

    st.sidebar.header("Real-time Controls")
    query_override = st.sidebar.text_input(
        "Override NewsAPI query",
        placeholder="Generative AI",
        help="Leave empty to use the default configured query.",
    )
    refresh_clicked = st.sidebar.button("Refresh data cache", key="refresh-cache")
    fetch_clicked = st.sidebar.button(
        "Ingest latest articles", type="primary", key="fetch-latest"
    )

    status_message: str | None = None
    if fetch_clicked:
        with st.spinner("Contacting NewsAPI + Azure AI..."):
            inserted_ids = get_pipeline().run(
                query=query_override or None,
            )
        load_dataframe.clear()
        status_message = (
            f"Ingested {len(inserted_ids)} articles at "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}."
        )
    elif refresh_clicked:
        load_dataframe.clear()

    df = load_dataframe()
    if status_message:
        st.success(status_message)

    if df.empty:
        st.warning("No documents found in Cosmos DB. Run the ingestion pipeline first.")
        return

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
