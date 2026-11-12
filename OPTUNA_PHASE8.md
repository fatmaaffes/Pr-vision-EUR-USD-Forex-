# Optimisation automatique LSTM/GRU — Phase 8 bis

La page `pages/phase_08_optuna.py` ajoute une étude Optuna **sans remplacer**
la phase 8 existante. Elle figure dans la navigation `app.py`.

## Installation

Depuis le dossier du projet :

```powershell
python -m pip install optuna
python -c "import torch, optuna; print(torch.__version__, optuna.__version__)"
python -m streamlit run app.py
```

PyTorch doit déjà fonctionner dans l'environnement Python sélectionné.
En cas de `WinError 126`, réparer l'installation de PyTorch avant d'exécuter l'étude.

## Mode d'emploi

1. Ouvrir **08 bis · Optuna LSTM/GRU** dans la barre latérale.
2. Choisir **LSTM**, puis le nombre d'essais et d'époques maximales.
3. Lancer la recherche et examiner la **MAE de validation**, la baseline naïve,
   les meilleurs hyperparamètres, l'historique des essais et les pertes.
4. Répéter pour **GRU**. Les exports CSV et les poids `.pt` sont disponibles.

## Protocole / limites

- Source : taux de référence EUR/USD de la Banque centrale européenne.
- Cible : prochaine publication BCE (J+1 en observations, pas en jours civils).
- Préparation univariée autonome : dates ordonnées, taux valides, pas d'interpolation.
- Découpage par cibles : 70 % TRAIN / 15 % VALIDATION / 15 % TEST réservé.
- Normalisation : moyenne et écart-type calculés sur TRAIN uniquement.
- Les fenêtres de validation peuvent contenir des observations antérieures
  de validation connues au moment de prédire la prochaine publication.
- Optuna minimise la MAE de validation exprimée en USD par EUR.
- Early stopping et pruning ne consultent pas TEST.
- Le test existant a déjà été consulté : il faut réserver un nouvel
  échantillon futur pour une nouvelle conclusion indépendante.
- Cette page n'injecte pas automatiquement ses résultats dans la phase 9.
  Sa préparation utilise une normalisation différente de celle de la phase 6 ;
  elle constitue une **expérience séparée**, non un remplacement silencieux
  des modèles déjà enregistrés dans Streamlit.

## État du dépôt GitHub

La branche `main` peut ne pas contenir les mises à jour locales apportées
aux phases 4–9 depuis des archives ZIP. Avant de synchroniser Git, vérifier
les modifications locales et ne pas écraser les pages déjà développées.
