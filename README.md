# AI Pulse News Tracker

AI Pulse News Tracker ingests the latest AI-related articles from NewsAPI, scores them with Azure AI Language, stores the enriched payload in Azure Cosmos DB, and exposes a Streamlit dashboard to explore the data.

## Features
- French-language AI news ingestion via NewsAPI (batch size configurable through environment variables).
- Azure Text Analytics sentiment scoring with confidence values retained per article.
- Cosmos DB persistence with automatic database/container provisioning and article `id` set to the source URL so re-ingests update in place.
- Streamlit dashboard showing KPIs, plots, and a latest-articles table backed by the Cosmos data.

## Project Layout
```
.
|-- app.py               # Streamlit entry point (thin wrapper around the package)
|-- news_analyzer.py     # CLI helper to run the ingestion pipeline once or continuously
|-- src/
|   |-- ai_pulse_tracker/
|       |-- config.py    # Settings loader + validation
|       |-- dashboard.py # Streamlit UI definition
|       |-- models.py    # Dataclasses shared across modules
|       |-- news.py      # NewsAPI client
|       |-- pipeline.py  # Orchestrates ingestion/analyze/persist flow
|       |-- sentiment.py # Azure AI Language wrapper
|       |-- storage.py   # Cosmos DB repository
|-- tests/               # pytest-based smoke tests
|-- .env.example         # Copy to .env and fill in your secrets
|-- pyproject.toml       # Packaging + dependency definition
|-- requirements.txt     # Runtime dependency mirror (optional)
```

## Prerequisites
- Python 3.10+
- Azure subscription with AI Language + Cosmos DB (Core SQL API)
- NewsAPI key

## Setup
1. Clone the repository and create a virtual environment.
2. Copy `.env.example` to `.env` and populate all keys.
3. Install dependencies (editable mode keeps `src/` on `PYTHONPATH`):
   ```bash
   pip install -e .[dev]
   ```

## Usage
### Run the ingestion pipeline once
Fetch and analyze the newest AI stories (optionally override the query term):
```bash
python news_analyzer.py --query "Generative AI"
```
- Need to reprocess older coverage? Append `--full-refresh` to ignore the incremental cursor or pass `--since 2024-04-01T00:00:00Z` to re-fetch articles after a specific timestamp.

### Launch the dashboard
```bash
streamlit run app.py
```
The Streamlit script loads the Cosmos DB container and visualizes article counts, sentiment distribution, and raw entries.

## Real-time Tracking
- **Continuous ingestion:** run `python news_analyzer.py --interval 120` to pull/analyze news every two minutes (Ctrl+C to stop). The CLI enforces a 30-second minimum cadence to protect the upstream APIs.
- **On-demand ingestion:** inside the Streamlit app, use the sidebar "Ingest latest articles" button to trigger a fresh NewsAPI pull + Azure analysis and refresh the dashboard cache immediately.
- The sidebar also exposes optional fields to supply an ISO datetime (to re-fetch after a custom point) or ignore the incremental cursor entirely, mirroring the CLI flags.
- Articles are deduplicated by URL (the Cosmos `id` equals the article URL) and normalized per-domain partition keys, so repeated runs update existing documents even if NewsAPI changes the source name.
- Each ingestion only requests stories published after the most recent one stored in Cosmos DB, keeping the “latest articles” panel focused on truly new coverage instead of repeating the same five headlines.

## Testing
Use pytest for the lightweight configuration tests:
```bash
pytest
```

## Configuration Reference
Environment variables (see `.env.example`):
- `AZURE_AI_ENDPOINT`, `AZURE_AI_KEY`
- `NEWS_API_KEY`
- `COSMOS_ENDPOINT`, `COSMOS_KEY`
- Optional overrides: `COSMOS_DATABASE`, `COSMOS_CONTAINER`, `NEWS_QUERY`, `NEWS_LANGUAGE`, `NEWS_BATCH_SIZE`

Missing required keys will raise a `SettingsError` before any API call is issued.
