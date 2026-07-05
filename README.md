# AI Pulse - Azure Sentiment Monitor

AI Pulse is a Streamlit dashboard and RAG assistant for monitoring AI news signals.
It ingests articles from NewsAPI, enriches them with Azure AI Language sentiment,
stores the results in Azure Cosmos DB, and exposes analytical views for trends,
companies, sources, sentiment, importance and weak signals.

The assistant combines:

- compact dashboard metrics;
- retrieved article evidence;
- LLM-generated synthesis with citations;
- deterministic dashboard fallback when the LLM provider is unavailable.

## Features

- News ingestion from NewsAPI with incremental cursor support.
- Azure AI Language sentiment analysis.
- AI relevance filtering to keep the dataset focused on AI / Generative AI.
- Keyword, topic, company and importance enrichment.
- Cosmos DB persistence with deduplication.
- Streamlit dashboard with KPIs, filters and visual analytics.
- RAG-style assistant using `dashboard_context` + retrieved articles.
- Groq/OpenAI-compatible LLM endpoint support through environment variables.
- Deterministic assistant fallback for rate limits or missing LLM credentials.

## Repository Layout

```text
.
|-- app.py                         # Streamlit entrypoint
|-- news_analyzer.py               # Ingestion CLI
|-- requirements.txt               # Runtime dependencies
|-- pyproject.toml                 # Package metadata and dev dependencies
|-- .env.example                   # Environment variable template
|-- .streamlit/
|   |-- config.toml                # Streamlit theme
|-- docs/
|   |-- AI_PRODUCT_ROADMAP.md      # Product roadmap
|-- scripts/
|   |-- check_setup.py             # Optional local API connectivity check
|-- src/
|   |-- ai_pulse_tracker/
|       |-- agent.py               # RAG assistant and LLM/fallback logic
|       |-- config.py              # Settings loader
|       |-- dashboard.py           # Streamlit dashboard UI
|       |-- embeddings.py          # Local deterministic embeddings
|       |-- enrichment.py          # Keyword/topic/company enrichment
|       |-- models.py              # Domain models
|       |-- news.py                # NewsAPI client
|       |-- pipeline.py            # Ingestion orchestration
|       |-- relevance.py           # AI relevance filtering
|       |-- retrieval.py           # Hybrid article retrieval
|       |-- sentiment.py           # Azure AI Language client
|       |-- storage.py             # Cosmos DB repository
|       |-- trends.py              # Keyword utilities
|-- tests/                         # Pytest suite
```

## Requirements

- Python 3.10+
- NewsAPI key
- Azure AI Language resource
- Azure Cosmos DB account using Core SQL API
- Optional: OpenAI-compatible LLM provider for assistant synthesis

## Installation

```bash
git clone <repo-url>
cd AI-Pulse-News-Tracker
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Create a local `.env` file from the template:

```bash
copy .env.example .env
```

Required variables:

```env
AZURE_AI_ENDPOINT=https://<your-ai-endpoint>.cognitiveservices.azure.com/
AZURE_AI_KEY=
NEWS_API_KEY=
COSMOS_ENDPOINT=https://<your-account>.documents.azure.com:443/
COSMOS_KEY=
COSMOS_DATABASE=NewsDatabase
COSMOS_CONTAINER=Analyses
NEWS_QUERY=("Generative AI" OR ChatGPT OR OpenAI OR LLM OR "AI agents" OR RAG OR Gemini OR Copilot OR Mistral)
NEWS_LANGUAGE=fr
NEWS_BATCH_SIZE=50
NEWS_MAX_LOOKBACK_DAYS=29
```

Optional LLM variables:

```env
AI_PULSE_LLM_API_KEY=
AI_PULSE_LLM_MODEL=gpt-4o-mini
AI_PULSE_LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
```

For Groq or another OpenAI-compatible provider, set:

```env
AI_PULSE_LLM_ENDPOINT=https://api.groq.com/openai/v1/chat/completions
AI_PULSE_LLM_MODEL=<groq-model-name>
AI_PULSE_LLM_API_KEY=<provider-api-key>
```

Never commit `.env` or local key files.

## Run Ingestion

Run one ingestion pass:

```bash
python news_analyzer.py
```

Override the query:

```bash
python news_analyzer.py --query "Generative AI"
```

Fetch articles from a specific date:

```bash
python news_analyzer.py --since 2024-04-01T00:00:00Z
```

Force a refresh without the incremental cursor:

```bash
python news_analyzer.py --full-refresh
```

Run continuously:

```bash
python news_analyzer.py --interval 120
```

## Run Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

- global filters by date, sentiment, source and importance;
- KPI cards;
- sentiment distribution and sentiment trend;
- importance analysis by company;
- topic and keyword trends;
- source and company donut charts;
- styled article explorer;
- AI Pulse Assistant for RAG-style analysis.

## Assistant Behavior

The assistant uses a hybrid context:

```text
user question
-> intent detection
-> compact dashboard_context
-> article retrieval
-> compact retrieved_evidence
-> LLM synthesis
-> cited analytical answer
```

It does not send the full dataframe or full article contents to the LLM.
Retrieved evidence is compact and capped to a small article set.

If the LLM is unavailable, rate-limited or not configured, the assistant returns a deterministic dashboard-based fallback instead of crashing.

## Utility Scripts

Check local API configuration and NewsAPI connectivity:

```bash
python scripts/check_setup.py
```

This script uses your local `.env` and should not be part of automated tests.

## Tests

```bash
python -m pytest tests
```

The test suite covers:

- configuration loading;
- ingestion and NewsAPI query behavior;
- enrichment, relevance and embeddings;
- retrieval scoring;
- dashboard context building;
- assistant LLM payload protection and fallback behavior.

## Security

- Keep API keys in `.env`.
- Keep `.env`, local key files, virtual environments and caches out of Git.
- Use `.env.example` only to document expected variables.
- Do not paste secrets into issues, commits or screenshots.

## Roadmap

See `docs/AI_PRODUCT_ROADMAP.md`.
