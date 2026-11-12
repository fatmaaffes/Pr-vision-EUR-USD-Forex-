# Dashboard utilisateur Forex EUR/USD — phase 11

## Objectif

Une page d'accueil orientée utilisateur qui distingue strictement :

1. **Prévision J+1 (prochaine publication BCE)** : taux naïf et GRU Optuna, calculés
   à partir des taux réellement connus.
2. **Régime observé** : écart-type des 20 derniers rendements en %, comparé au
   seuil médian déterminé sur TRAIN.
3. **Simulation du risque de change** : hypothèses de marchés calme/agité,
   stress EUR/USD ± une volatilité et Monte Carlo sur 1, 5, 10 ou 20 publications.

La volatilité ne prédit pas la direction, et le scénario calm/agité ne modifie
jamais artificiellement la prévision produite par le GRU.

## Pages

- Phase 11 : **dashboard final métier** et page d'accueil de `app.py`.
- Phase 8 bis : entraînement Optuna LSTM/GRU (technique).
- Phase 9 bis : comparaison des erreurs sur validation selon les régimes (technique).

## Démarrage

```powershell
cd F:\Forex-DL-Risk-Prediction
git status
git pull origin main
python -m pip install -r requirements.txt
python -m pytest tests/test_dashboard_forex.py tests/test_volatility_regimes.py -q
python -m streamlit run app.py
```

Si `git status` montre des changements locaux conflictuels, les sauvegarder
et les examiner avant `git pull`. Le dépôt GitHub peut ne pas contenir
les versions locales les plus récentes des pages 4–9.

## Modèle GRU disponible sans upload

La phase 8 bis enregistre automatiquement le meilleur modèle dans :

`artifacts/forex_models/gru_optuna_best.pt`

La phase 11 le charge automatiquement ; si une session Optuna existe encore,
elle peut l'utiliser directement. Le fichier est ignoré par Git ; sur un serveur
sans stockage persistant, il faut prévoir une solution de stockage durable.

**Important :** une optimisation exécutée AVANT cette modification n'est pas
automatiquement enregistrée sur disque. Si les anciens poids sont encore en
mémoire, on peut les exporter ; sinon, il faut réexécuter Optuna.

Lorsque le GRU n'est pas disponible ou que les données TRAIN ne correspondent
plus, l'application montre seulement la prévision naïve et un diagnostic ;
elle n'invente aucune prévision GRU.

## Méthodologie

- Le GRU entraîné sur les anciennes données peut prévoir J+1 à partir des
  derniers taux connus sans consulter les taux futurs ni se réentraîner.
- La recommandation « naïf pour calme / GRU pour agité » reste **exploratoire**
  : l'avantage GRU historique en régime agité était très faible et non validé
  indépendamment.
- Monte Carlo utilise 5 000 tirages de chocs logarithmiques indépendants,
  volatilité constante par scénario et aucun effet politique modélisé.
- L'horizon Monte Carlo (plusieurs publications BCE) n'est **pas** une
  prévision multi-jours du GRU.
- Les taux de référence BCE ne comprennent ni frais ni spreads bancaires.

## Validation

`tests/test_dashboard_forex.py` vérifie les calculs de scénario,
les quantiles, le comportement causal de la volatilité et le seuil TRAIN.
Le rendu Streamlit et le chargement PyTorch des modèles doivent être
validés sur la machine qui exécute l'application.
