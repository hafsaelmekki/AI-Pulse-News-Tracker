# AI Pulse - Product Roadmap

This roadmap turns AI Pulse into a stronger portfolio project for Data Scientist,
AI Engineer, Data/AI Consultant, LLM, RAG, AI agent, automation and API-oriented roles.

## 1. Article Enrichment

Goal: enrich each ingested article with reusable business signals.

- Extract keywords from article titles and descriptions.
- Compute an importance score from sentiment confidence, source/title quality and keyword richness.
- Store topics, companies, relevance signals and embeddings in Cosmos DB.

Status: implemented.

## 2. Trend Dashboard

Goal: make the Streamlit dashboard business-oriented and demo-ready.

- Show KPIs, sentiment trends, topic trends and importance evolution.
- Add filters by date, sentiment, source and minimum importance.
- Highlight top articles, sources, companies and weak signals.

Status: implemented.

## 3. RAG Assistant

Goal: provide an assistant that combines dashboard metrics and retrieved article evidence.

- Detect user intent.
- Build compact `dashboard_context`.
- Retrieve relevant articles.
- Send compact dashboard + retrieved evidence to an LLM.
- Cite article evidence with stable citation ids.
- Fall back to deterministic dashboard answers when the LLM is unavailable.

Status: implemented.

## 4. Semantic Retrieval

Goal: query articles by meaning, not only keywords.

- Generate deterministic embeddings for article content.
- Rank results with hybrid lexical + vector scoring.
- Keep retrieved evidence compact to avoid large LLM payloads.

Status: implemented locally. A dedicated vector database remains a future option.

## 5. Deployment

Goal: make the project easy to run and demonstrate.

- Add Docker support.
- Add deployment documentation for Streamlit and Azure.
- Optionally expose ingestion/search through a FastAPI layer.

Status: planned.
