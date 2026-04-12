import os
from dotenv import load_dotenv
import requests

# Charger les variables du fichier .env
load_dotenv()

azure_key = os.getenv("AZURE_AI_KEY")
news_key = os.getenv("NEWS_API_KEY")

print("--- Vérification de la configuration ---")

if azure_key:
    print("✅ Clé Azure détectée.")
else:
    print("❌ Clé Azure manquante dans le .env")

if news_key:
    # Test rapide de NewsAPI
    url = f"https://newsapi.org/v2/everything?q=AI&pageSize=1&apiKey={news_key}"
    response = requests.get(url)
    if response.status_code == 200:
        print("✅ NewsAPI fonctionne parfaitement !")
    else:
        print(f"❌ Erreur NewsAPI : {response.status_code}")
else:
    print("❌ Clé NewsAPI manquante dans le .env")
