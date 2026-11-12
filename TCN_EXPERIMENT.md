# Nouveau modèle Deep Learning : TCN (Temporal Convolutional Network)

Le projet compare désormais trois architectures sur les taux EUR/USD BCE :

- **LSTM** et **GRU** : réseaux récurrents avec mémoire et portes.
- **TCN** : convolutions temporelles **causales et dilatées**, blocs résiduels.

La TCN implémentée reçoit les mêmes fenêtres historiques normalisées de
taux que LSTM/GRU et prédit le taux de la prochaine publication BCE (J+1).

## Comment lancer l'expérience

1. Dans Streamlit, ouvrir **08 bis · Optuna LSTM/GRU/TCN**.
2. Choisir **TCN**, puis 10 essais, 30 époques max, patience 7.
3. Lancer Optuna, lire la **MAE VALIDATION** et la comparer au naïf.
4. Consulter **09 bis · Régimes de volatilité** : la nouvelle ligne
   **TCN Optuna** figure dans chaque régime si les poids ont été entraînés
   et chargés. Le graphique affiche des barres côte à côte.
5. Conserver les métriques dans le rapport : globalement et selon le régime
   de volatilité (d'après 20 rendements précédant chaque cible).

La sélection des hyperparamètres utilise seulement TRAIN et VALIDATION.
Le TEST historique a déjà été consulté pour les premières expériences ;
il ne faut pas s'en servir pour décider si TCN devient le meilleur modèle.
Une évaluation chronologique indépendante sera nécessaire.

## Sauvegarde / compatibilité

La page Optuna sauvegarde le meilleur TCN après entraînement :

`artifacts/forex_models/tcn_optuna_best.pt`

Aucun nouveau package Deep Learning n'est nécessaire : TCN est écrit avec
PyTorch. Les anciens poids LSTM et GRU restent compatibles (noms des couches
`recurrent` et `head` conservés).

**L'utilisateur final ne change pas automatiquement de modèle** : le
dashboard phase 11 conserve sa politique exploratoire existante. Le
TCN sert d'abord à la **comparaison scientifique** et ne devra être retenu
pour une prévision opérationnelle que si ses performances sont confirmées.

## Tests

```powershell
python -m pytest tests/test_tcn_architectures.py -q
python -m streamlit run app.py
```

Les tests vérifient les tenseurs de sortie, les gradients, la sérialisation,
et l'absence de fuite temporelle des convolutions (un point futur modifié
ne doit pas altérer les représentations passées).

Attention : le test unitaire sur données synthétiques ne remplace pas
l'entraînement réel sur les données BCE, que l'utilisateur devra lancer.
