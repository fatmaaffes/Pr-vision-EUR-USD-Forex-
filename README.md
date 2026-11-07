# Forex EUR/USD — Version Streamlit multipage

## Installation dans `F:\Forex-DL-Risk-Prediction`

1. **Sauvegarder** le `app.py` existant sous `app_backup.py`.
2. Décompresser le ZIP et copier **son contenu** dans `F:\Forex-DL-Risk-Prediction` (pas le dossier parent).
3. Accepter le remplacement de `app.py`. Les fichiers existants non concernés ne sont pas supprimés.
4. Dans le terminal VS Code, depuis le dossier du projet, lancer :

```powershell
python -m streamlit run app.py
```

Le menu latéral montre maintenant 12 phases distinctes.

## Structure

- `app.py` : navigation latérale avec `st.Page` / `st.navigation`.
- `src/bce.py` : chargement de l'API BCE, partagé et mis en cache.
- `pages/phase_01.py` : cadrage métier et simulateur de conversion.
- `pages/phase_02.py` : dimensions, colonnes, qualité, statuts, période et dates récentes (ancien code conservé).
- `pages/phase_03.py` : courbe historique et statistiques descriptives (ancien code conservé).
- `pages/phase_04.py` à `phase_12.py` : pages réservées au développement futur.

La source de données n'a pas changé :
`https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata`

## Remarques

- Ne pas conserver une autre déclaration `st.set_page_config` dans les fichiers de phase ; celle du routeur suffit.
- Les statistiques et les données BCE sont préservées, mais les sections sont réparties entre les phases 2 et 3.
- Le cache `st.cache_data(ttl=3600)` reste en place.
- Aucun nettoyage ni entraînement ne sont réalisés par cette modification.
- Les taux du simulateur de la phase 1 sont fictifs.
