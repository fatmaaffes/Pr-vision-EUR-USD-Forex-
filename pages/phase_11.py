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
      .fx-regime {
        border-left: 4px solid #5aafaf; background: rgba(70,130,148,.11);
        padding: 14px 18px; border-radius: 12px; margin: 10px 0 18px;
        line-height: 1.55;
      }
      .fx-panel {
        border: 1px solid rgba(145,164,190,.28);
        border-radius: 18px; padding: 22px 24px; margin: 12px 0;
        background: rgba(118,142,170,.075);
      }
      .fx-panel-primary {
        border-color: rgba(87,169,174,.52);
        background: linear-gradient(115deg, rgba(47,123,133,.22), rgba(90,120,153,.06));
      }
      .fx-kicker {color: #93bfcb; font-size: .84rem; font-weight: 650;
        letter-spacing: .035em; margin-bottom: 7px;}
      .fx-panel-title {font-size: 1.14rem; font-weight: 620; margin-bottom: 8px;}
      .fx-value {font-size: clamp(2rem, 5vw, 2.6rem); font-weight: 750;
        letter-spacing: -.025em; line-height: 1.15; margin: 8px 0 12px;}
      .fx-muted {font-size: .9rem; opacity: .84; line-height: 1.6;}
      .fx-panel-title, .fx-value, .fx-regime {overflow-wrap: anywhere;}
      @media (max-width: 800px) {.fx-panel {padding: 17px 18px;}}
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
st.subheader(f"Situation du marché — dernière publication BCE : {latest_date:%d/%m/%Y}")
m1, m2 = st.columns(2)
m1.metric(
    "Dernier taux EUR/USD",
    f"{latest_rate:.4f}",
    help="1 euro correspond à ce nombre de dollars américains (référence BCE).",
)
m2.metric("Volatilité des 20 derniers rendements", f"{volatility:.3f} %")
st.markdown(
    f'<div class="fx-regime"><strong>Marché actuellement '
    f'{"agité" if is_high else "calme"}.</strong> '
    f'Les variations récentes sont '
    f'{"élevées" if is_high else "modérées"} '
    f'par rapport au seuil historique de {threshold:.3f} %.</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Dernière publication BCE reçue : {latest_date:%d/%m/%Y}. "
    "La volatilité est calculée sur les rendements déjà connus, "
    "y compris celui de la dernière publication."
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

# Ces deux valeurs sont calculées à partir de la situation connue AUJOURD'HUI.
# Le régime ne transforme pas le GRU en modèle conditionnel à un choc futur.
if is_high and gru_forecast is not None:
    main_forecast = gru_forecast
else:
    main_forecast = naive_forecast


def forecast_card(kicker: str, title: str, value: str, explanation: str, primary=False):
    """Carte métier : aucun jargon ML dans le premier niveau de lecture."""
    style = " fx-panel-primary" if primary else ""
    st.markdown(
        f'<div class="fx-panel{style}">'
        f'<div class="fx-kicker">{kicker}</div>'
        f'<div class="fx-panel-title">{title}</div>'
        f'<div class="fx-value">{value}</div>'
        f'<div class="fx-muted">{explanation}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


with forecast_tab:
    st.subheader("Prévision pour la prochaine publication BCE")
    st.write(
        "Nous partons du dernier taux connu et du niveau d'agitation "
        "observé aujourd'hui. Deux approches donnent deux lectures possibles."
    )

    if not is_high:
        forecast_card(
            "PRÉVISION MISE EN AVANT · MARCHÉ CALME",
            "Avec la faible volatilité observée aujourd'hui",
            f"{naive_forecast:.4f} USD/EUR",
            "Dans ce contexte, notre méthode de prévision privilégie la stabilité "
            "du dernier taux publié. Elle a été légèrement plus précise dans "
            "les périodes calmes de notre analyse historique.",
            primary=True,
        )
        if gru_forecast is not None:
            forecast_card(
                "AUTRE ESTIMATION · À SURVEILLER SI LE MARCHÉ S'AGITE",
                "Si les conditions de marché deviennent plus instables",
                f"{gru_forecast:.4f} USD/EUR",
                "Voici la seconde estimation calculée aujourd'hui à partir des "
                "mêmes données connues. Elle provient de l'approche qui avait "
                "obtenu une erreur légèrement inférieure dans les périodes agitées. "
                "Elle ne tient pas compte d'un événement futur encore inconnu.",
            )
        else:
            forecast_card(
                "AUTRE ESTIMATION · MARCHÉ PLUS AGITÉ",
                "Si la volatilité augmente demain",
                "À calculer",
                "La seconde estimation nécessite un modèle préalablement enregistré. "
                "Nous ne remplaçons pas sa prévision manquante par un chiffre inventé.",
            )
    else:
        if gru_forecast is not None:
            forecast_card(
                "PRÉVISION MISE EN AVANT · MARCHÉ AGITÉ",
                "Avec la forte volatilité observée aujourd'hui",
                f"{gru_forecast:.4f} USD/EUR",
                "Cette estimation provient de la méthode qui avait légèrement "
                "mieux résisté aux périodes agitées dans notre analyse historique. "
                "Son avantage n'est pas encore confirmé sur de nouvelles données.",
                primary=True,
            )
        else:
            forecast_card(
                "MARCHÉ AGITÉ · ESTIMATION À CONFIRMER",
                "La prévision adaptée aux périodes agitées est indisponible",
                "À calculer",
                "Le modèle correspondant n'est pas enregistré sur cet appareil. "
                "La référence de stabilité reste disponible ci-dessous.",
                primary=True,
            )
        forecast_card(
            "AUTRE ESTIMATION · MARCHÉ CALME",
            "Si le marché retrouve des variations modérées",
            f"{naive_forecast:.4f} USD/EUR",
            "Cette estimation de stabilité utilise le dernier taux connu. "
            "Elle est affichée comme point de comparaison, et non comme "
            "une prévision conditionnelle de la volatilité future.",
        )

    # Le scénario de forte volatilité est un intervalle d'hypothèses,
    # PAS la prédiction du modèle avec un événement politique fictif.
    high_stress = stressed_rates(latest_rate, context["high_sigma"])
    low_end = high_stress["Dépréciation EUR"]
    high_end = high_stress["Appréciation EUR"]
    st.markdown("#### Et si une annonce provoque de fortes variations demain ?")
    st.write(
        f"À titre de **simulation**, une variation d'environ "
        f"±{context['high_sigma']:.3f} % autour du dernier taux BCE donnerait "
        f"une plage indicative de **{low_end:.4f} à {high_end:.4f} USD/EUR**."
    )
    st.caption(
        "Cette plage n'est pas une prévision du modèle. "
        "Une hausse de volatilité peut entraîner une hausse OU une baisse du taux ; "
        "nous ne connaissons pas à l'avance l'effet d'une annonce politique."
    )
    st.info(
        "La situation du jour sert à mettre une estimation en avant. "
        "Les deux approches sont calculées avec les informations connues aujourd'hui ; "
        "le comportement réel du marché demain reste incertain. "
        "La comparaison des approches demeure expérimentale."
    )

    if gru_forecast is None:
        with st.expander("Pourquoi manque-t-il une estimation ? · Informations techniques"):
            st.write(
                "Le modèle de réseau neuronal n'est pas encore chargé. "
                "Ouvre la phase 8 bis pour enregistrer ses poids, puis reviens ici."
            )
            st.page_link(
                "pages/phase_08_optuna.py",
                label="Ouvrir la préparation du modèle", icon="🔬",
            )
            st.code(gru_error or "Aucun modèle enregistré")

    st.write("#### Évolution récente de l'EUR/USD")
    history = rates.tail(100)[["TIME_PERIOD", "OBS_VALUE"]].copy()
    history.columns = ["Date", "EUR/USD"]
    history_chart = (
        alt.Chart(history)
        .mark_line(color="#438fb5", strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", title="Date BCE"),
            y=alt.Y("EUR/USD:Q", title="USD par EUR", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                alt.Tooltip("EUR/USD:Q", title="EUR/USD", format=".4f"),
            ],
        )
    )
    st.altair_chart(history_chart.properties(height=235), width="stretch")
    st.caption(
        "J+1 signifie la prochaine publication BCE, pas nécessairement le "
        "prochain jour du calendrier. Les taux affichés sont indicatifs."
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

        **Et le TCN ?** Un réseau convolutif temporel est désormais testé dans
        les pages techniques 8 bis et 9 bis. Il ne remplacera pas le modèle
        proposé à l'utilisateur sans démontrer une amélioration robuste.

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
