import os
import uuid
import requests
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey

load_dotenv()

# --- Configuration ---
# Azure AI
AI_KEY = os.getenv("AZURE_AI_KEY")
AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
# News API
NEWS_KEY = os.getenv("NEWS_API_KEY")
# Cosmos DB
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "NewsDatabase"
CONTAINER_NAME = "Analyses"

# --- Fonctions ---


def get_cosmos_container():
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.create_database_if_not_exists(id=DATABASE_NAME)
    container = db.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path="/source"),
        offer_throughput=400  # Option gratuite
    )
    return container


def fetch_ai_news(query="Generative AI"):
    url = f"https://newsapi.org/v2/everything?q={query}&language=fr&pageSize=5&apiKey={NEWS_KEY}"
    response = requests.get(url)
    return response.json().get("articles", []) if response.status_code == 200 else []


def analyze_and_save():
    ai_client = TextAnalyticsClient(AI_ENDPOINT, AzureKeyCredential(AI_KEY))
    container = get_cosmos_container()
    articles = fetch_ai_news()

    print(f"🔍 Analyse de {len(articles)} articles...")

    for art in articles:
        text = f"{art['title']}. {art['description']}"
        sentiment_response = ai_client.analyze_sentiment(documents=[text])[0]

        # Structure de donnée propre (JSON)
        doc = {
            "id": str(uuid.uuid4()),
            "source": art["source"]["name"],
            "title": art["title"],
            "sentiment": sentiment_response.sentiment,
            "confidence": {
                "pos": sentiment_response.confidence_scores.positive,
                "neu": sentiment_response.confidence_scores.neutral,
                "neg": sentiment_response.confidence_scores.negative
            },
            "url": art["url"],
            "date": art["publishedAt"]
        }

        # Enregistrement dans Cosmos DB
        container.upsert_item(doc)
        print(
            f"✅ Article sauvé : {art['title'][:50]}... [{sentiment_response.sentiment}]")


if __name__ == "__main__":
    analyze_and_save()
