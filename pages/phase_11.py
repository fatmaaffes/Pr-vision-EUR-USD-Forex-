"""Dashboard métier EUR/USD : prévisions J+1, volatilité et scénarios financiers.

Les calculs des modèles sont distincts des hypothèses de stress / Monte Carlo.
Les modèles sont expérimentaux et les taux BCE sont indicatifs.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.bce import get_data_or_stop
from src.volatility_regimes import clean_rates
from src.dashboard_forex import (
    volatility_context, stressed_rates, monte_carlo_costs,
)
from src.dashboard_models import load_checkpoint, predict_next


st.markdown(
    """
    <style>
      .block-container {max-width: 1150px; padding-top: 1.6rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {
        background: rgba(122, 142, 175, .085);
        border: 1px solid rgba(138, 154, 181, .23);
        padding: 16px 18px;
        border-radius: 15px;
      }
      div[data-testid="stMetricLabel"] {font-size: .88rem;}
      div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("FOREX INSIGHT  /  TABLEAU DE BORD FINANCIER")
st.title("EUR/USD — Prévisions & risque de change")
st.write(
    "Un aperçu simple des prévisions de la prochaine publication BCE, "
    "du régime de volatilité observé et des conséquences pour une facture en dollars."
)

try:
    rates = clean_rates(get_data_or_stop())
    values = rates["OBS_VALUE"].to_numpy(dtype=float)
    context = volatility_context(values, window=20)
except (KeyError, ValueError, TypeError, IndexError) as exc:
    st.error(f"Impossible de préparer le dashboard : {exc}")
    st.stop()

latest_date = pd.Timestamp(rates.iloc[-1]["TIME_PERIOD"]).date()
latest_rate = context["last_rate"]
volatility = context["recent"]
threshold = context["threshold"]
is_high = context["regime"] == "Forte"
regime_label = "Forte volatilité" if is_high else "Faible volatilité"

# Les deux prévisions partent de la même information connue aujourd'hui.
naive_forecast = latest_rate
gru_forecast = None
gru_source = ""
gru_error = ""

checkpoint, loaded_from = load_checkpoint(
    "GRU", st.session_state.get("forex_optuna_GRU")
)
if checkpoint is not None:
    try:
        gru_forecast = predict_next(checkpoint, "GRU", rates)
        gru_source = loaded_from
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        gru_error = str(exc)
else:
    gru_error = loaded_from

st.divider()
st.subheader("Le marché aujourd'hui")
m1, m2, m3 = st.columns(3)
m1.metric("Dernier taux de référence", f"{latest_rate:.4f}", help="1 EUR vaut ce nombre de dollars US.")
m2.metric("Volatilité récente (20 rendements)", f"{volatility:.3f} %")
m3.metric("Régime observé", regime_label)
st.caption(
    f"Dernier taux BCE connu : {latest_date:%d/%m/%Y} · "
    f"Seuil de volatilité appris sur TRAIN : {threshold:.3f} % · "
    "Volatilité basée uniquement sur les publications déjà disponibles."
)

today_utc = pd.Timestamp.now(tz="UTC").date()
if (today_utc - latest_date).days > 7:
    st.warning(
        "La dernière publication BCE reçue remonte à plus de 7 jours. "
        "Les prévisions ne doivent pas être présentées comme actualisées."
    )

forecast_tab, risk_tab, explainer_tab = st.tabs(
    ["📈 Prévisions J+1", "💶 Scénarios & facture USD", "ℹ️ Comment lire ces résultats"]
)

with forecast_tab:
    st.subheader("Deux approches, deux estimations")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Naïf · maintien du dernier taux", f"{naive_forecast:.4f}")
        st.caption("Prévision de persistance : le prochain taux = dernier taux BCE.")
    with p2:
        if gru_forecast is not None:
            st.metric(
                "GRU · Deep Learning optimisé",
                f"{gru_forecast:.4f}",
                delta=f"{(gru_forecast / latest_rate - 1) * 100:+.3f} % vs aujourd'hui",
                delta_color="off",
            )
            st.caption(f"Calcul réel du réseau entraîné · {gru_source}.")
        else:
            st.metric("GRU · Deep Learning", "Non disponible")
            st.caption("Aucune prévision GRU inventée : le modèle doit être entraîné.")

    if is_high and gru_forecast is not None:
        displayed_model = "GRU"
        main_forecast = gru_forecast
        interpretation = (
            "La volatilité récente est **élevée**. Nous mettons en avant le GRU "
            "à titre expérimental, car il a légèrement mieux résisté au régime "
            "volatil sur notre validation. Le gain était faible et non confirmé "
            "sur une période indépendante."
        )
    else:
        displayed_model = "Naïf"
        main_forecast = naive_forecast
        interpretation = (
            "La volatilité récente est **faible** : la prévision naïve reste "
            "notre référence la plus robuste sur la validation."
            if not is_high else
            "La volatilité récente est **élevée**, mais le GRU n'est pas chargé. "
            "Nous utilisons temporairement la référence naïve."
        )

    st.info(
        f"**Lecture du jour — {displayed_model} mis en avant : "
        f"{main_forecast:.4f} USD/EUR.**\n\n{interpretation}"
    )
    st.caption(
        "Il ne s'agit pas d'une certitude ni d'un conseil de trading. "
        "La volatilité récente ne permet pas à elle seule de connaître "
        "la direction du prochain taux."
    )

    if gru_forecast is None:
        st.warning(
            "Pour afficher la prévision GRU réelle : ouvrir « 08 bis · Optuna LSTM/GRU », "
            "optimiser le GRU. La nouvelle version enregistre ensuite automatiquement ses poids."
        )
        st.page_link(
            "pages/phase_08_optuna.py", label="Ouvrir l'optimisation GRU",
            icon="🔬",
        )
        with st.expander("Diagnostic de chargement du GRU"):
            st.code(gru_error or "Aucun modèle")
    else:
        st.write("### Évolution récente du taux BCE")
        history = rates.tail(100)[["TIME_PERIOD", "OBS_VALUE"]].copy()
        history.columns = ["Date", "EUR/USD"]
        line = (
            alt.Chart(history)
            .mark_line(color="#438fb5", strokeWidth=2.5)
            .encode(
                x=alt.X("Date:T", title="Date de publication BCE"),
                y=alt.Y("EUR/USD:Q", title="USD par EUR", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                    alt.Tooltip("EUR/USD:Q", title="EUR/USD", format=".4f"),
                ],
            )
        )
        st.altair_chart(line.properties(height=255), width="stretch")
        st.caption(
            "Historique réalisé ; les deux prévisions J+1 sont affichées dans les cartes "
            "ci-dessus. La date exacte de la prochaine publication peut dépendre du calendrier BCE."
        )

with risk_tab:
    st.subheader("Si le marché devient plus calme ou plus agité…")
    st.write(
        "Ici, tu explores des **hypothèses financières**. "
        "Elles ne sont pas des prédictions politiques ni des taux GRU recalculés."
    )

    left, right = st.columns([1, 1])
    with left:
        risk_regime = st.radio(
            "Scénario de volatilité à étudier",
            ["Marché calme", "Marché agité"],
            index=1 if is_high else 0,
            horizontal=True,
        )
        multiple = st.slider(
            "Intensité relative du scénario", 0.5, 3.0, 1.0, 0.25,
            help="Augmente ou réduit les fluctuations hypothétiques, pas le taux prédit par le GRU.",
        )
    with right:
        usd_amount = st.number_input(
            "Facture à régler (USD)", min_value=100.0, value=100000.0,
            step=5000.0, format="%.2f"
        )
        horizon = st.select_slider(
            "Horizon de simulation (publications BCE)",
            options=[1, 5, 10, 20],
            value=1,
        )

    base_sigma = (
        context["low_sigma"] if risk_regime == "Marché calme"
        else context["high_sigma"]
    )
    hypothetical_sigma = base_sigma * multiple
    st.caption(
        f"Hypothèse de volatilité journalière : {hypothetical_sigma:.3f} % "
        f"(médiane historique du régime choisi sur TRAIN × {multiple:.2f})."
    )

    # Référence de simulation explicitement fixée à la prévision du modèle
    # mis en avant, sans prétendre prédire un effet causal de la volatilité.
    anchor_choice = st.radio(
        "Taux de départ des scénarios",
        ["Dernier taux BCE", "Prévision mise en avant"],
        index=0,
        horizontal=True,
    )
    anchor = latest_rate if anchor_choice == "Dernier taux BCE" else main_forecast
    scenarios = stressed_rates(anchor, hypothetical_sigma)
    results = pd.DataFrame([
        {
            "Hypothèse J+1": name,
            "EUR/USD simulé": rate,
            "Coût de la facture (EUR)": usd_amount / rate,
            "Écart coût vs taux de départ (EUR)": usd_amount / rate - usd_amount / anchor,
        }
        for name, rate in scenarios.items()
    ])
    st.write("### Trois scénarios de change — un jour de publication")
    st.dataframe(
        results.style.format({
            "EUR/USD simulé": "{:.4f}",
            "Coût de la facture (EUR)": "{:,.2f}",
            "Écart coût vs taux de départ (EUR)": "{:+,.2f}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "La volatilité change la largeur du scénario ; elle n'indique ni hausse ni baisse. "
        "Les trois taux simulés ne sont pas des prédictions produites par LSTM/GRU."
    )

    st.write("### Monte Carlo — exposition au risque de change")
    mc = monte_carlo_costs(
        usd_amount=usd_amount, anchor=anchor,
        sigma_percent=hypothetical_sigma, horizon=horizon,
        n_paths=5000, seed=42,
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Coût médian simulé", f"{mc['p50']:,.0f} €")
    q2.metric("Scénario défavorable P95", f"{mc['p95']:,.0f} €")
    q3.metric("Coût au taux de départ", f"{mc['today']:,.0f} €")
    chart_data = pd.DataFrame({"Coût EUR": mc["costs"]})
    histogram = (
        alt.Chart(chart_data)
        .mark_bar(color="#438fb5")
        .encode(
            x=alt.X("Coût EUR:Q", bin=alt.Bin(maxbins=35), title="Coût de la facture (EUR)"),
            y=alt.Y("count():Q", title="Nombre de simulations"),
            tooltip=[alt.Tooltip("count():Q", title="Simulations")],
        )
    )
    st.altair_chart(histogram.properties(height=250), width="stretch")
    st.caption(
        f"5 000 trajectoires hypothétiques sur {horizon} publication(s) BCE ; "
        "rendements aléatoires indépendants, volatilité constante et distribution "
        "normale des chocs logarithmiques. P95 est un quantile simulé, "
        "pas une borne garantie ou une VaR réglementaire."
    )
    st.download_button(
        "Télécharger les scénarios (CSV)",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="forex_scenarios_facture.csv",
        mime="text/csv",
    )

with explainer_tab:
    st.markdown(
        """
        **Pourquoi deux prévisions ?** Le naïf suppose que le prochain taux sera
        identique au dernier. Le GRU cherche des relations dans l'historique
        des taux. Tous deux utilisent l'information disponible au moment de prévoir.

        **À quoi sert la volatilité ?** Les 20 derniers rendements BCE mesurent
        l'amplitude récente des variations. La comparaison au seuil TRAIN
        identifie un régime *actuel* de marché, et non celui de demain.

        **Pourquoi le GRU est-il mis en avant en forte volatilité ?**
        Notre analyse exploratoire sur VALIDATION suggère un gain très faible
        dans ce régime. Cela ne garantit aucune amélioration future :
        l'hypothèse doit être confirmée sur une période indépendante.

        **Que représente la simulation ?** Le scénario calme/agité et Monte Carlo
        modélisent différentes amplitudes de variation. Ni l'actualité politique
        ni le futur niveau de volatilité ne sont ici prédits.

        **Quel horizon ?** Les modèles naïf/GRU prévoient uniquement la prochaine
        **publication BCE**. L'horizon 5/10/20 de Monte Carlo concerne seulement
        la simulation de risque, pas une prévision multi-jours du GRU.

        **Limites métier :** taux de référence BCE, non négociable en tant que tel ;
        spread bancaire, frais, taux forward et couvertures ne sont pas inclus.
        """
    )
    st.page_link(
        "pages/phase_09_regimes.py",
        label="Consulter l'analyse technique par volatilité",
        icon="🌡️",
    )

st.divider()
st.caption(
    "Source : Banque centrale européenne (EUR/USD). Prototype académique "
    "Deep Learning & FinTech — scénarios indicatifs, sans conseil financier."
)
