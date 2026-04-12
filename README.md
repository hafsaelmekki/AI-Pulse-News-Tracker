# AI Pulse News Tracker

AI Pulse News Tracker ingests the latest AI-related articles from NewsAPI, runs Azure AI Language sentiment analysis, stores the enriched payload in Azure Cosmos DB, and visualizes the results inside a Streamlit dashboard.

## Features
- Pulls French-language AI articles from NewsAPI in batches of five.
- Scores each headline + description with Azure AI Text Analytics and persists scores/confidence values in Cosmos DB.
- Streamlit dashboard shows sentiment distribution, optimism score, and the latest analyzed articles.

## Requirements
- Python 3.10+
- Azure subscription with:
  - Azure AI Language resource (endpoint + key)
  - Azure Cosmos DB (Core SQL API)
- NewsAPI account/key

## Environment Variables
Store the following keys inside a `.env` file at the project root:

```
AZURE_AI_ENDPOINT=
AZURE_AI_KEY=
NEWS_API_KEY=
COSMOS_ENDPOINT=
COSMOS_KEY=
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Fill the `.env` file with your Azure/NewsAPI credentials.

## Usage
### 1. Collect & analyze news
Run the ETL/analyzer script to pull the most recent articles and push them into Cosmos DB:
```bash
python news_analyzer.py
```

### 2. Launch the dashboard
Start the Streamlit UI to explore the stored analyses:
```bash
streamlit run app.py
```
The dashboard reads from the same Cosmos DB container (`NewsDatabase/Analyses`).

## Project Structure
```
app.py             # Streamlit dashboard
news_analyzer.py   # Data ingestion + sentiment analysis pipeline
requirements.txt   # Python dependencies
.env               # Local secrets (never commit)
```

## Notes
- `news_analyzer.py` automatically creates the Cosmos DB database/container if they do not exist.
- Cached data in `app.py` reduces Cosmos DB reads for 10 minutes (`@st.cache_data`).
- When deploying, configure the same environment variables in your hosting environment.
