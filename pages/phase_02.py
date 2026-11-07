"""Phase 2 — Compréhension de la structure et de la qualité du dataset BCE.

Cette page explique les observations et les statuts sans nettoyer les données.
"""

import pandas as pd
import streamlit as st

from src.bce import get_data_or_stop

st.title("📥 Phase 2 — Collecte et compréhension du dataset BCE")
st.caption("Source officielle : Banque centrale européenne (BCE) · EUR/USD quotidien")
st.link_button(
    "Voir la série officielle BCE",
    "https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A",
)

df = get_data_or_stop()

# 1. Dimension du jeu de données
st.subheader("1. Dimensions du dataset")
st.write("Nombre de lignes :", df.shape[0])
st.write("Nombre de colonnes :", df.shape[1])

# 2. Noms des colonnes
st.subheader("2. Colonnes disponibles")
st.write(df.columns.tolist())
with st.expander("💡 Pourquoi 32 colonnes pour un seul taux EUR/USD ?"):
    st.markdown(
        """
        Les **32 colonnes ne représentent pas 32 variables économiques**.
        La BCE fournit le taux et des **métadonnées** qui expliquent l'observation.

        - `TIME_PERIOD` : date de l'observation.
        - `OBS_VALUE` : taux EUR/USD de référence, en USD pour 1 EUR.
        - `OBS_STATUS` : statut de l'observation (par exemple `A` ou `H`).
        - `FREQ`, `CURRENCY`, `CURRENCY_DENOM`, `EXR_TYPE` : identité de la série.
        - Les autres colonnes : informations et métadonnées statistiques.

        Nous étudierons leur signification plus en détail avant de choisir les
        variables à conserver pour la modélisation.
        """
    )

# 3. Aperçu source
st.subheader("3. Premières observations")
st.dataframe(df.head(20), use_container_width=True, hide_index=True)

# 4. Qualité générale, sans imputation
st.subheader("4. Types et valeurs manquantes")
info = pd.DataFrame({
    "Type": df.dtypes.astype(str),
    "Valeurs manquantes": df.isnull().sum(),
})
st.dataframe(info, use_container_width=True)
st.caption(
    "Une valeur manquante dans une métadonnée facultative n'est pas forcément "
    "un taux EUR/USD manquant. Pour les taux, regardons OBS_VALUE et OBS_STATUS."
)

# 5. Sens des statuts selon la nomenclature statistique BCE/SDMX
st.subheader("5. Analyse des valeurs manquantes et des statuts")

significations = {
    "A": "Normal value — valeur normale publiée",
    "H": "Missing value due to holiday or weekend — absence liée à un jour férié ou week-end",
}
statuts = (
    df["OBS_STATUS"]
    .fillna("Non renseigné")
    .value_counts()
    .rename_axis("OBS_STATUS")
    .reset_index(name="Nombre")
)
statuts["Signification"] = statuts["OBS_STATUS"].map(significations).fillna(
    "Autre statut : vérifier le dictionnaire BCE"
)

st.write("**Répartition des statuts des observations :**")
st.dataframe(statuts, use_container_width=True, hide_index=True)

st.markdown(
    """
    **Comment interpréter les codes ?**

    - **`A` = valeur normale** : un taux EUR/USD est disponible pour la date.
    - **`H` = jour férié ou week-end** : la valeur de taux n'est pas renseignée
      pour cette date. Une absence n'est **pas** un taux de zéro.

    Attention : le **`A` de `OBS_STATUS`** ne correspond pas à la même notion que
    le **`A` de `EXR_SUFFIX`** (variante statistique de la série).
    """
)

missing = df[df["OBS_VALUE"].isna()]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Lignes du dataset", f"{len(df):,}".replace(",", " "))
with col2:
    st.metric("Taux disponibles", f"{df['OBS_VALUE'].notna().sum():,}".replace(",", " "))
with col3:
    st.metric("Taux manquants", f"{len(missing):,}".replace(",", " "))

st.write("**Dates dont OBS_VALUE est manquant :**")
st.dataframe(
    missing[["TIME_PERIOD", "OBS_STATUS", "OBS_VALUE"]],
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Nous ne supprimons ni ne remplaçons ces observations à cette étape. "
    "Les week-ends peuvent aussi être totalement absents du dataset."
)

# 6. Début et fin de l'historique
st.subheader("6. Période couverte par le dataset")
dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
dates_valides = dates.dropna()
if not dates_valides.empty:
    premiere_date = dates_valides.min()
    derniere_date = dates_valides.max()
    duree_annees = (derniere_date - premiere_date).days / 365.25

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Première date", premiere_date.strftime("%d/%m/%Y"))
    with col2:
        st.metric("Dernière date", derniere_date.strftime("%d/%m/%Y"))
    with col3:
        st.metric("Durée historique", f"{duree_annees:.1f} ans")

    st.markdown(
        """
        **Que signifie cette période ?**

        - **Première date** : début de notre historique disponible.
        - **Dernière date** : observation la plus récente présente **dans les données chargées**.
        - **Durée historique** : distance entre ces deux dates, **pas** le nombre
          de jours ayant un taux publié.
        - **Fréquence `D` (daily)** : observations quotidiennes **lorsqu'une
          observation est prévue**, pas forcément tous les jours civils.

        La BCE publie normalement les taux de référence les jours ouvrés
        concernés, sauf notamment les jours de fermeture TARGET.
        """
    )
    st.caption(
        "La dernière date peut être antérieure à aujourd'hui, selon le calendrier "
        "de publication BCE et le cache de chargement (1 heure)."
    )
else:
    st.warning("Aucune date interprétable dans TIME_PERIOD.")

# 7. Dernières observations
st.subheader("7. Dernières observations BCE")
dernieres_observations = (
    df[["TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]]
    .sort_values("TIME_PERIOD", ascending=False)
    .head(10)
)
st.dataframe(dernieres_observations, use_container_width=True, hide_index=True)

st.caption("Phase 2 : compréhension des données uniquement. Aucun nettoyage n'a été appliqué.")

st.subheader("9. Dictionnaire des colonnes principales")

dictionnaire_colonnes = pd.DataFrame([
    {
        "Colonne dans la base": "TIME_PERIOD",
        "Nom simple": "Date",
        "Signification": "Date de l'observation du taux.",
        "Utilité pour le projet": "Permet d'ordonner les données dans le temps."
    },
    {
        "Colonne dans la base": "OBS_VALUE",
        "Nom simple": "Taux EUR/USD",
        "Signification": "Valeur du taux de change EUR/USD publiée par la BCE.",
        "Utilité pour le projet": "C'est la variable principale que nous allons analyser et prédire."
    },
    {
        "Colonne dans la base": "OBS_STATUS",
        "Nom simple": "Statut de l'observation",
        "Signification": "Indique si la valeur est normale (A) ou manquante pour jour férié/week-end (H).",
        "Utilité pour le projet": "Permet de comprendre et contrôler les valeurs manquantes."
    },
    {
        "Colonne dans la base": "FREQ",
        "Nom simple": "Fréquence",
        "Signification": "Fréquence des données. Ici 'D' signifie Daily (quotidien).",
        "Utilité pour le projet": "Confirme que la série est journalière."
    },
    {
        "Colonne dans la base": "CURRENCY",
        "Nom simple": "Devise cotée",
        "Signification": "Devise de cotation. Ici USD.",
        "Utilité pour le projet": "Aide à comprendre la paire de devises."
    },
    {
        "Colonne dans la base": "CURRENCY_DENOM",
        "Nom simple": "Devise de base",
        "Signification": "Devise de base. Ici EUR.",
        "Utilité pour le projet": "Aide à comprendre la paire de devises."
    },
    {
        "Colonne dans la base": "EXR_TYPE",
        "Nom simple": "Type de change",
        "Signification": "Type du taux de change. Ici SP00 = Spot.",
        "Utilité pour le projet": "Permet de comprendre la nature du taux observé."
    },
    {
        "Colonne dans la base": "KEY",
        "Nom simple": "Clé de la série",
        "Signification": "Identifiant complet de la série BCE.",
        "Utilité pour le projet": "Permet d’identifier précisément la série statistique."
    },
    {
        "Colonne dans la base": "EXR_SUFFIX",
        "Nom simple": "Suffixe BCE",
        "Signification": "Variante statistique utilisée par la BCE.",
        "Utilité pour le projet": "Métadonnée descriptive, utile pour documenter la série."
    },
])

st.dataframe(dictionnaire_colonnes, use_container_width=True, hide_index=True)

st.info(
    "À retenir : pour notre projet, les colonnes les plus importantes sont "
    "TIME_PERIOD, OBS_VALUE et OBS_STATUS. "
    "Les autres colonnes servent surtout à documenter et comprendre la série BCE."
)
with st.expander("Comment lire ce tableau ?"):
    st.markdown("""
- **TIME_PERIOD** : indique la date de chaque observation.
- **OBS_VALUE** : contient le taux EUR/USD à cette date.
- **OBS_STATUS** : permet de savoir si le taux est normal (`A`) ou absent pour raison de calendrier (`H`).
- Les autres colonnes sont surtout des **métadonnées**, c’est-à-dire des informations qui décrivent la série.
    
👉 Pour le Deep Learning, nous n'utiliserons pas forcément toutes les colonnes.  
Les plus importantes au début sont surtout **la date**, **le taux**, et **le statut**.
""")
    
st.subheader("10. Contrôle des dates et des doublons")

# Convertir les dates
dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")

# Vérifier les doublons de dates
nb_doublons = dates.duplicated().sum()

# Construire toutes les dates du calendrier
calendrier = pd.date_range(
    start=dates.min(),
    end=dates.max(),
    freq="D"
)

# Dates présentes dans le dataset
dates_presentes = pd.DatetimeIndex(dates.dropna())

# Identifier les dates sans aucune ligne
dates_absentes = calendrier.difference(dates_presentes)

# Jours de semaine uniquement (lundi à vendredi)
jours_semaine = pd.bdate_range(dates.min(), dates.max())
jours_semaine_absents = jours_semaine.difference(dates_presentes)

col1, col2, col3 = st.columns(3)

col1.metric("Dates en doublon", nb_doublons)
col2.metric("Dates absentes du calendrier", len(dates_absentes))
col3.metric("Lundi-vendredi absents", len(jours_semaine_absents))

st.write("Exemples de dates absentes (lundi-vendredi) :")
st.dataframe(pd.DataFrame({
    "Date": jours_semaine_absents[:20]
}), hide_index=True)

st.info(
    "Les jours absents ne sont pas automatiquement des erreurs. "
    "Le calendrier lundi-vendredi inclut certains jours fériés "
    "où la BCE ne publie pas de taux."
)
