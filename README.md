# AI Pulse News Tracker

AI Pulse News Tracker collecte des articles récents sur l'intelligence artificielle, analyse leur sentiment avec Azure AI Language, les stocke dans Azure Cosmos DB, puis les affiche dans un tableau de bord Streamlit.

Le projet sert à suivre rapidement les tendances de l'actualité IA, en particulier les articles francophones, avec une lecture synthétique des volumes, sources et sentiments.

## Aperçu

- Collecte d'articles via NewsAPI.
- Analyse de sentiment via Azure AI Language.
- Stockage des résultats enrichis dans Azure Cosmos DB.
- Tableau de bord Streamlit avec indicateurs, graphiques et filtres.
- Mode ponctuel ou suivi continu avec intervalle configurable.

## Fonctionnalités

- Ingestion incrémentale des nouveaux articles pour éviter les doublons.
- Première exécution avec historique automatique sur les 30 derniers jours.
- Déduplication par URL dans Cosmos DB.
- Filtres par période, sentiment et mots-clés dans le dashboard.
- Commandes CLI pour relancer une collecte complète ou depuis une date précise.

## Architecture

```text
.
|-- app.py                 # Point d'entrée du dashboard Streamlit
|-- news_analyzer.py       # CLI pour lancer l'ingestion
|-- src/
|   |-- ai_pulse_tracker/
|       |-- config.py      # Chargement et validation de la configuration
|       |-- dashboard.py   # Interface Streamlit
|       |-- models.py      # Modèles de données
|       |-- news.py        # Client NewsAPI
|       |-- pipeline.py    # Orchestration collecte/analyse/stockage
|       |-- sentiment.py   # Client Azure AI Language
|       |-- storage.py     # Accès Azure Cosmos DB
|-- tests/                 # Tests pytest
|-- .env.example           # Modèle de configuration
|-- pyproject.toml         # Packaging et dépendances
|-- requirements.txt       # Dépendances runtime
```

## Prérequis

- Python 3.10 ou plus récent.
- Une clé NewsAPI.
- Un service Azure AI Language.
- Un compte Azure Cosmos DB avec l'API Core SQL.

## Installation

```bash
git clone <url-du-repo>
cd AI-Pulse-News-Tracker
python -m venv venv
venv\Scripts\activate
pip install -e .[dev]
```

## Configuration

Copier le fichier d'exemple :

```bash
copy .env.example .env
```

Puis remplir les variables dans `.env` :

```env
AZURE_AI_ENDPOINT=https://<your-ai-endpoint>.cognitiveservices.azure.com/
AZURE_AI_KEY=
NEWS_API_KEY=
COSMOS_ENDPOINT=https://<your-account>.documents.azure.com:443/
COSMOS_KEY=
COSMOS_DATABASE=NewsDatabase
COSMOS_CONTAINER=Analyses
NEWS_QUERY=Generative AI
NEWS_LANGUAGE=fr
NEWS_BATCH_SIZE=5
```

Ne jamais publier le fichier `.env` sur GitHub. Le dépôt doit seulement contenir `.env.example`.

## Utilisation

Lancer une collecte ponctuelle :

```bash
python news_analyzer.py
```

Utiliser une requête spécifique :

```bash
python news_analyzer.py --query "Generative AI"
```

Relancer une collecte depuis une date :

```bash
python news_analyzer.py --since 2024-04-01T00:00:00Z
```

Ignorer le curseur incrémental et récupérer le dernier lot :

```bash
python news_analyzer.py --full-refresh
```

Lancer un suivi continu toutes les deux minutes :

```bash
python news_analyzer.py --interval 120
```

## Dashboard

Lancer l'interface Streamlit :

```bash
streamlit run app.py
```

Le dashboard permet de :

- visualiser le nombre d'articles analysés ;
- identifier le sentiment dominant ;
- comparer les sources ;
- filtrer par période, sentiment et mots-clés ;
- déclencher une nouvelle ingestion depuis la barre latérale.

## Tests

```bash
pytest
```

## Roadmap

- Ajouter plus de sources d'actualité.
- Ajouter des alertes email ou Discord.
- Ajouter un résumé automatique des articles.
- Ajouter un export CSV ou JSON depuis le dashboard.
- Déployer le dashboard sur Streamlit Community Cloud ou Azure.

## Sécurité

- Garder les clés API dans `.env`.
- Vérifier que `.env` est bien ignoré par Git.
- Utiliser `.env.example` uniquement pour documenter les variables attendues.
