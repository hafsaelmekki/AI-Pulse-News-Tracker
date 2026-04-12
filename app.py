import streamlit as st
import pandas as pd
import os
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
import plotly.express as px

# Configuration
load_dotenv()
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "NewsDatabase"
CONTAINER_NAME = "Analyses"

# Initialisation Cosmos
client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# --- UI Streamlit ---
st.set_page_config(page_title="AI Pulse Tracker", layout="wide")
st.title("🌐 AI-Pulse: Analyseur de Sentiment en Temps Réel")
st.markdown(
    "Ce dashboard analyse les dernières news sur l'IA via **Azure AI Services**.")

# 1. Récupération des données


# Garde les données en cache 10 min pour économiser les requêtes
@st.cache_data(ttl=600)
def load_data():
    query = "SELECT * FROM c"
    items = list(container.query_items(
        query=query, enable_cross_partition_query=True))
    return pd.DataFrame(items)


df = load_data()

if not df.empty:
    # 2. Indicateurs clés (Metrics)
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles analysés", len(df))
    sentiment_counts = df['sentiment'].value_counts()
    col2.metric("Dominant", sentiment_counts.index[0].upper())
    col3.metric("Sources uniques", len(df['source'].unique()))

    # 3. Graphiques
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Répartition des sentiments")
        fig = px.pie(df, names='sentiment', color='sentiment',
                     color_discrete_map={'positive': '#00CC96', 'neutral': '#636EFA', 'negative': '#EF553B'})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Scores de confiance moyens")
        # On extrait les scores du dictionnaire 'confidence'
        df['pos_score'] = df['confidence'].apply(lambda x: x['pos'])
        avg_pos = df['pos_score'].mean()
        st.progress(avg_pos, text=f"Optimisme global : {avg_pos:.0%}")
        st.info("Ce score est calculé par le modèle NLP d'Azure AI Language.")

    # 4. Table des données brutes
    st.subheader("Dernières analyses")
    st.dataframe(
        df[['date', 'source', 'title', 'sentiment', 'url']], use_container_width=True)

else:
    st.warning(
        "Aucune donnée trouvée dans Cosmos DB. Lancez d'abord news_analyzer.py !")
