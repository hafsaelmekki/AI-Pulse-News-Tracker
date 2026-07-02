# AI Pulse News Tracker - Product Roadmap

This roadmap turns AI Pulse News Tracker into a stronger portfolio project for Data Scientist, AI Engineer, Data/AI Consultant, LLM, RAG, AI agent, automation and API-oriented roles.

## Step 1 - Article Enrichment

Goal: enrich each ingested article with reusable business signals.

- Extract keywords from article titles and descriptions.
- Compute an importance score from sentiment confidence, source/title quality and keyword richness.
- Store these signals in Cosmos DB for dashboarding and future retrieval.

## Step 2 - Trend Dashboard

Goal: make the Streamlit dashboard more business-oriented.

- Show top keywords and recurring themes.
- Add source and topic trend views.
- Add article importance filtering.

Status: implemented with keyword trends, source coverage and importance-based filtering.

## Step 3 - Generative AI Summaries

Goal: produce short and useful summaries.

- Add automatic article summaries.
- Add daily or weekly trend summaries.
- Keep summaries reproducible and stored with each article.

Status: article-level summaries are implemented with local deterministic generation and stored with each ingested article. LLM-based daily or weekly summaries remain planned.

## Step 4 - Semantic Search and RAG

Goal: query articles by meaning, not only keywords.

- Generate embeddings for article content.
- Store vectors in a vector database or a compatible search layer.
- Add semantic search over collected articles.

Status: RAG-ready local retrieval is implemented with ranked article search over titles, summaries, descriptions, keywords and importance scores. Embeddings and vector storage remain planned.

## Step 5 - AI Agent

Goal: provide an assistant for trend questions.

- Answer questions such as "What are the AI trends this week?"
- Retrieve relevant articles before answering.
- Cite sources from the stored article database.

## Step 6 - Deployment

Goal: make the project easy to run and demonstrate.

- Add Docker support.
- Add a FastAPI layer for ingestion and search endpoints.
- Add deployment documentation for Streamlit and cloud hosting.
