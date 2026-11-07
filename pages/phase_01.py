"""Phase 1 — Compréhension métier, avec simulateur financier pédagogique."""

import streamlit as st

st.title("🏦 Phase 1 — Cadrage métier")
st.write(
    "Notre client fictif est une entreprise européenne qui doit payer "
    "un fournisseur américain en dollars."
)
st.info(
    "Objectif : comprendre l'exposition au risque de change avant de prédire "
    "le taux EUR/USD. Les taux ci-dessous sont hypothétiques."
)

st.subheader("Simulation d'une facture en USD")
montant_usd = st.number_input(
    "Montant à payer (USD)", min_value=0.0, value=100000.0, step=1000.0
)
taux_initial = st.number_input(
    "Taux EUR/USD initial", min_value=0.0001, value=1.10, step=0.01, format="%.4f"
)
taux_scenario = st.number_input(
    "Taux EUR/USD futur simulé", min_value=0.0001, value=1.05, step=0.01, format="%.4f"
)

cout_initial = montant_usd / taux_initial
cout_scenario = montant_usd / taux_scenario
difference = cout_scenario - cout_initial

col1, col2, col3 = st.columns(3)
col1.metric("Coût au taux initial", f"{cout_initial:,.2f} €")
col2.metric("Coût au taux simulé", f"{cout_scenario:,.2f} €")
col3.metric("Surcoût / économie", f"{difference:+,.2f} €", delta_color="inverse", delta=f"{difference:+,.2f} €")

st.latex(r"\text{Coût en euros} = \frac{\text{Montant en USD}}{\text{EUR/USD}}")
st.caption(
    "Simulation pédagogique hors spread, frais bancaires et couverture. "
    "Elle ne constitue pas une prévision du taux dans 30 jours."
)
