"""Point d'entrée Streamlit : navigation des phases du projet EUR/USD."""

import streamlit as st

st.set_page_config(
    page_title="Forex EUR/USD — Projet DL & Risque",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.caption("Projet académique · Deep Learning & Finance")

pages = {
    "Comprendre et explorer": [
        st.Page("pages/phase_01.py", title="01 · Cadrage métier", icon="🏦", url_path="phase-01"),
        st.Page("pages/phase_02.py", title="02 · Collecte et dataset BCE", icon="📥", url_path="phase-02"),
        st.Page("pages/phase_03.py", title="03 · Exploration EDA", icon="📊", url_path="phase-03"),
    ],
    "Préparer et modéliser": [
        st.Page("pages/phase_04.py", title="04 · Prétraitement", icon="🧹", url_path="phase-04"),
        st.Page("pages/phase_05.py", title="05 · Feature Engineering", icon="🛠️", url_path="phase-05"),
        st.Page("pages/phase_06.py", title="06 · Préparation des séquences", icon="🧮", url_path="phase-06"),
        st.Page("pages/phase_07.py", title="07 · Baselines et ML", icon="📏", url_path="phase-07"),
        st.Page("pages/phase_08.py", title="08 · Deep Learning", icon="🧠", url_path="phase-08"),
        st.Page("pages/phase_08_optuna.py", title="08 bis · Optuna LSTM/GRU", icon="🔬", url_path="phase-08-optuna"),
        st.Page("pages/phase_09.py", title="09 · Évaluation", icon="🔬", url_path="phase-09"),
        st.Page("pages/phase_09_regimes.py", title="09 bis · Régimes de volatilité", icon="🌡️", url_path="phase-09-regimes"),
    ],
    "Appliquer et livrer": [
        st.Page("pages/phase_10.py", title="10 · Risque de change", icon="💶", url_path="phase-10"),
        st.Page("pages/phase_11.py", title="11 · Dashboard utilisateur", icon="🖥️", url_path="phase-11", default=True),
        st.Page("pages/phase_12.py", title="12 · Validation et livraison", icon="✅", url_path="phase-12"),
    ],
}

page = st.navigation(pages, position="sidebar")
page.run()
