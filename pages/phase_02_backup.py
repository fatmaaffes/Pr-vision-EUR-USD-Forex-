"""Phase 2 — Structure et qualité du dataset BCE (ancien app.py conservé)."""

import pandas as pd
import streamlit as st

from src.bce import ECB_URL, get_data_or_stop

st.title("📥 Phase 2 — Collecte et compréhension du dataset BCE")
st.caption("Source officielle : Banque centrale européenne (BCE) · EUR/USD quotidien")
st.link_button("Voir la source BCE", "https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A")

df = get_data_or_stop()

st.subheader("1. Dimensions du dataset")
st.write("Nombre de lignes :", df.shape[0])
st.write("Nombre de colonnes :", df.shape[1])

st.subheader("2. Colonnes disponibles")
st.write(df.columns.tolist())

st.subheader("3. Premières observations")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("4. Types et valeurs manquantes")
info = pd.DataFrame({
    "Type": df.dtypes.astype(str),
    "Valeurs manquantes": df.isnull().sum(),
})
st.dataframe(info, use_container_width=True)

st.subheader("5. Analyse des valeurs manquantes et des statuts")
st.write("Répartition des statuts :")
st.dataframe(
    df["OBS_STATUS"].value_counts(dropna=False).rename_axis("OBS_STATUS").reset_index(name="Nombre"),
    use_container_width=True,
    hide_index=True,
)

missing = df[df["OBS_VALUE"].isna()]
st.write(f"Observations dont OBS_VALUE est manquant : {len(missing)}")
st.dataframe(
    missing[["TIME_PERIOD", "OBS_STATUS", "OBS_VALUE"]],
    use_container_width=True,
)

st.subheader("6. Période couverte par le dataset")
dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
if dates.notna().any():
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Première date", dates.min().strftime("%d/%m/%Y"))
    with col2:
        st.metric("Dernière date", dates.max().strftime("%d/%m/%Y"))
else:
    st.warning("Aucune date interprétable dans TIME_PERIOD.")

st.subheader("7. Dernières observations BCE")
dernieres_observations = (
    df[["TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]]
    .sort_values("TIME_PERIOD", ascending=False)
    .head(10)
)
st.dataframe(dernieres_observations, use_container_width=True, hide_index=True)

st.caption("Aucune valeur manquante n'est supprimée ou imputée à cette étape.")
