import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forex EUR/USD", layout="wide")
st.title("📊 Exploration du dataset EUR/USD")

# Récupération des données depuis la BCE
@st.cache_data(ttl=3600)
def load_data():
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "EXR/D.USD.EUR.SP00.A?format=csvdata"
    )
    return pd.read_csv(url)

try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

# Informations générales
st.subheader("1. Dimensions du dataset")
st.write("Nombre de lignes :", df.shape[0])
st.write("Nombre de colonnes :", df.shape[1])

# Colonnes disponibles
st.subheader("2. Colonnes disponibles")
st.write(df.columns.tolist())

# Aperçu
st.subheader("3. Premières observations")
st.dataframe(df.head(20), use_container_width=True)

# Types et valeurs manquantes
st.subheader("4. Types et valeurs manquantes")
info = pd.DataFrame({
    "Type": df.dtypes.astype(str),
    "Valeurs manquantes": df.isnull().sum()
})
st.dataframe(info)

# Visualisation du taux EUR/USD
st.subheader("5. Évolution historique EUR/USD")

historique = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
historique["TIME_PERIOD"] = pd.to_datetime(
    historique["TIME_PERIOD"]
)
historique = historique.set_index("TIME_PERIOD").sort_index()

st.line_chart(historique["OBS_VALUE"])

st.subheader("6. Analyse des valeurs manquantes")

# Statistiques des taux EUR/USD
st.write("Statistiques descriptives :")
st.dataframe(df["OBS_VALUE"].describe())

# Statuts des observations
st.write("Répartition des statuts :")
st.dataframe(
    df["OBS_STATUS"].value_counts(dropna=False)
)

# Observations sans taux
missing = df[df["OBS_VALUE"].isna()]

st.write("Observations avec taux manquant :")
st.dataframe(
    missing[["TIME_PERIOD", "OBS_STATUS", "OBS_VALUE"]]
)

# 7. Période couverte par le dataset
st.subheader("7. Période couverte par le dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Première date",
        historique.index.min().strftime("%d/%m/%Y")
    )

with col2:
    st.metric(
        "Dernière date",
        historique.index.max().strftime("%d/%m/%Y")
    )

# 8. Dernières observations
st.subheader("8. Dernières observations BCE")

dernieres_observations = (
    df[["TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]]
    .sort_values("TIME_PERIOD")
    .tail(10)
    .sort_values("TIME_PERIOD", ascending=False)
)

st.dataframe(
    dernieres_observations,
    use_container_width=True,
    hide_index=True
)

