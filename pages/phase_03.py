"""Phase 3 - Exploration progressive des taux EUR/USD (BCE)."""

import pandas as pd
import streamlit as st

from src.bce import get_data_or_stop

st.title("📊 Phase 3 — Exploration des données (EDA)")
st.write(
    "Objectif : observer et comprendre le comportement historique du taux EUR/USD "
    "avant tout prétraitement ou entraînement."
)

# Le chargement est partagé avec les autres pages de l'application.
df = get_data_or_stop()

# Copie de travail pour la visualisation : les données brutes ne sont pas modifiées.
serie = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
serie["TIME_PERIOD"] = pd.to_datetime(serie["TIME_PERIOD"], errors="coerce")
serie["OBS_VALUE"] = pd.to_numeric(serie["OBS_VALUE"], errors="coerce")
serie = serie.dropna(subset=["TIME_PERIOD"]).sort_values("TIME_PERIOD")

st.subheader("1. Observer l'évolution historique EUR/USD")
st.write(
    "Chaque point représente un taux de référence BCE publié pour une date. "
    "La courbe relie les observations pour faciliter la lecture."
)

periode = st.selectbox(
    "Période à afficher",
    ["12 derniers mois", "3 derniers mois", "5 dernières années", "Tout l'historique"],
    index=0,
)

fin = serie["TIME_PERIOD"].max()
if periode == "3 derniers mois":
    debut = fin - pd.DateOffset(months=3)
elif periode == "12 derniers mois":
    debut = fin - pd.DateOffset(years=1)
elif periode == "5 dernières années":
    debut = fin - pd.DateOffset(years=5)
else:
    debut = serie["TIME_PERIOD"].min()

periode_df = serie.loc[serie["TIME_PERIOD"] >= debut].copy()
valeurs_disponibles = periode_df.dropna(subset=["OBS_VALUE"])

if valeurs_disponibles.empty:
    st.warning("Aucun taux disponible pour cette période.")
    st.stop()

st.line_chart(
    valeurs_disponibles.set_index("TIME_PERIOD")["OBS_VALUE"],
    use_container_width=True,
)
st.caption(
    "Le graphique n'affiche pas les taux absents (NaN). "
    "Ceci est un choix d'affichage ; le dataset original n'est pas nettoyé. "
    "La série BCE est quotidienne aux dates de publication, pas à chaque jour civil."
)

st.subheader("2. Comprendre les niveaux du taux")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Taux le plus ancien affiché", f"{valeurs_disponibles['OBS_VALUE'].iloc[0]:.4f}")
col2.metric("Dernier taux affiché", f"{valeurs_disponibles['OBS_VALUE'].iloc[-1]:.4f}")
col3.metric("Minimum de la période", f"{valeurs_disponibles['OBS_VALUE'].min():.4f}")
col4.metric("Maximum de la période", f"{valeurs_disponibles['OBS_VALUE'].max():.4f}")

st.write("Statistiques descriptives de la période sélectionnée :")
st.dataframe(
    valeurs_disponibles["OBS_VALUE"].describe().rename("EUR/USD"),
    use_container_width=True,
)

with st.expander("Comment interpréter ces statistiques ?"):
    st.markdown(
        """
- **count** : nombre de taux disponibles dans la période affichée.
- **mean** : moyenne des taux observés (ce n'est pas une prévision).
- **std** : dispersion des niveaux du taux autour de leur moyenne ; **ce n'est pas encore notre mesure de volatilité financière**.
- **min / max** : taux le plus bas / le plus haut observé sur cette période.
- **25 % / 50 % / 75 %** : quartiles ; **50 %** correspond à la médiane.
        """
    )

st.subheader("3. Questions à se poser sur la courbe")
st.markdown(
    """
1. Sur la période affichée, le taux monte-t-il, baisse-t-il ou oscille-t-il ?
2. Existe-t-il des périodes où les mouvements paraissent plus importants ?
3. Comment la lecture change-t-elle entre **3 mois**, **1 an** et **tout l'historique** ?
    """
)
st.info(
    "Nous commencerons par l'interprétation du graphique. "
    "Les rendements, la mesure de la volatilité et la stationnarité seront étudiés ensuite."
)
# Étape 3.2 — Distribution des taux EUR/USD
st.subheader("4. Distribution des taux EUR/USD")

histogramme = pd.cut(
    valeurs_disponibles["OBS_VALUE"],
    bins=20
).value_counts(sort=False)

histogramme.index = histogramme.index.astype(str)

st.bar_chart(histogramme, use_container_width=True)

st.info(
    "Axe horizontal : intervalles EUR/USD. "
    "Axe vertical : nombre d'observations."
)

# Étape 3.3 — Rendements journaliers
st.subheader("5. Analyse des rendements EUR/USD")

st.write(
    "Le rendement mesure la variation en pourcentage du taux "
    "entre deux observations BCE successives."
)

rendements_df = (
    valeurs_disponibles[["TIME_PERIOD", "OBS_VALUE"]]
    .copy()
    .sort_values("TIME_PERIOD")
)

rendements_df["Rendement (%)"] = (
    rendements_df["OBS_VALUE"].pct_change(fill_method=None) * 100
)

# Première observation sans rendement précédent
rendements_df = rendements_df.dropna(subset=["Rendement (%)"])

st.line_chart(
    rendements_df.set_index("TIME_PERIOD")["Rendement (%)"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Variations positives",
    int((rendements_df["Rendement (%)"] > 0).sum())
)

col2.metric(
    "Variations négatives",
    int((rendements_df["Rendement (%)"] < 0).sum())
)

col3.metric(
    "Variations nulles",
    int((rendements_df["Rendement (%)"] == 0).sum())
)

st.write("Derniers rendements calculés :")
st.dataframe(
    rendements_df.tail(10),
    use_container_width=True,
    hide_index=True
)

st.info(
    "Rendement positif : EUR/USD augmente. "
    "Rendement négatif : EUR/USD diminue. "
    "Les variations sont calculées entre deux observations "
    "BCE successives."
)

# Étape 3.4 — Volatilité financière
st.subheader("6. Volatilité historique EUR/USD")

st.write(
    "La volatilité mesure la dispersion des rendements. "
    "Nous étudions ici la volatilité journalière et son "
    "évolution sur 20 observations successives."
)

# Écart-type des rendements sur la période choisie
volatilite = rendements_df["Rendement (%)"].std()

st.metric(
    "Volatilité journalière historique",
    f"{volatilite:.4f} %"
)

# Volatilité glissante sur 20 observations
volatilite_glissante = (
    rendements_df.set_index("TIME_PERIOD")["Rendement (%)"]
    .rolling(window=20)
    .std()
)

st.write("Évolution de la volatilité sur 20 observations :")
st.line_chart(volatilite_glissante)

st.info(
    "Courbe haute : rendements plus dispersés sur les "
    "20 dernières observations. "
    "Courbe basse : rendements moins dispersés. "
    "La volatilité n'indique pas la direction future du taux."
)

# Étape 3.5 — Stationnarité
from statsmodels.tsa.stattools import adfuller

st.subheader("7. Étude de la stationnarité — Test ADF")

st.write(
    "Nous comparons la stationnarité des taux EUR/USD "
    "et celle de leurs rendements."
)

# Séries à analyser sur la période sélectionnée
taux = valeurs_disponibles["OBS_VALUE"].dropna()
rendements = rendements_df["Rendement (%)"].dropna()

def calculer_adf(serie):
    resultat = adfuller(serie, autolag="AIC")
    return resultat[1]  # p-value

if len(taux) >= 30 and len(rendements) >= 30:
    p_taux = calculer_adf(taux)
    p_rendements = calculer_adf(rendements)

    resultats_adf = pd.DataFrame({
        "Série": ["Taux EUR/USD", "Rendements (%)"],
        "p-value ADF": [p_taux, p_rendements],
        "Interprétation": [
            "Racine unitaire rejetée" if p_taux < 0.05
            else "Racine unitaire non rejetée",
            "Racine unitaire rejetée" if p_rendements < 0.05
            else "Racine unitaire non rejetée"
        ]
    })

    st.dataframe(
        resultats_adf,
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("Pas assez d'observations pour cette analyse.")

st.info(
    "Le test ADF examine l'hypothèse de racine unitaire. "
    "Une p-value inférieure à 0,05 permet de rejeter "
    "cette hypothèse, mais ne prouve pas à elle seule "
    "que toutes les propriétés de la série sont stationnaires."
)

# Étape 3.5b — Autocorrélation
from statsmodels.tsa.stattools import acf

st.subheader("8. Autocorrélation des taux et rendements EUR/USD")

st.write(
    "L'autocorrélation mesure la relation entre une série "
    "et ses propres observations passées."
)

nb_lags = 20

taux_acf = (
    valeurs_disponibles.sort_values("TIME_PERIOD")["OBS_VALUE"]
    .dropna()
)

rendements_acf = (
    rendements_df.sort_values("TIME_PERIOD")["Rendement (%)"]
    .dropna()
)

if len(taux_acf) > nb_lags + 1 and len(rendements_acf) > nb_lags + 1:

    # Autocorrélation des niveaux du taux
    acf_taux = acf(taux_acf, nlags=nb_lags, fft=True)

    # Autocorrélation des rendements
    acf_rendements = acf(
        rendements_acf,
        nlags=nb_lags,
        fft=True
    )

    resultats_acf = pd.DataFrame({
        "Lag": range(1, nb_lags + 1),
        "Taux EUR/USD": acf_taux[1:],
        "Rendements (%)": acf_rendements[1:]
    }).set_index("Lag")

    st.write("### Autocorrélation des taux EUR/USD")
    st.bar_chart(resultats_acf["Taux EUR/USD"])

    st.write("### Autocorrélation des rendements")
    st.bar_chart(resultats_acf["Rendements (%)"])

    st.write("### Coefficients calculés")
    st.dataframe(resultats_acf.round(4))

else:
    st.warning("Pas assez d'observations pour calculer l'ACF.")

st.info(
    "Lag 1 = observation précédente. "
    "Lag 5 = cinq observations auparavant. "
    "Une autocorrélation élevée ne garantit pas "
    "une bonne capacité de prédiction."
)
