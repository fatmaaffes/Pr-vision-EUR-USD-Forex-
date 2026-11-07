"""Récupération centralisée de la série EUR/USD de la BCE."""

import pandas as pd
import streamlit as st

ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/"
    "EXR/D.USD.EUR.SP00.A?format=csvdata"
)


@st.cache_data(ttl=3600, show_spinner="Chargement des taux BCE…")
def load_data() -> pd.DataFrame:
    """Lit le fichier CSV de la BCE sans modifier les valeurs sources."""
    return pd.read_csv(ECB_URL)


def get_data_or_stop() -> pd.DataFrame:
    """Affiche une erreur lisible si l'API BCE est inaccessible."""
    try:
        return load_data()
    except Exception as error:
        st.error(f"Impossible de charger les données BCE : {error}")
        st.stop()
