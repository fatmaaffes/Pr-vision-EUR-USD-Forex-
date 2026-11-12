"""Optuna LSTM/GRU/TCN EUR/USD : validation chronologique, sans exploiter TEST."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from src.bce import get_data_or_stop

st.title("🔬 Phase 8 bis — Optimisation LSTM / GRU / TCN")
st.write(
    "Recherche d'hyperparamètres pour une prédiction **J+1** (prochaine publication BCE). "
    "On entraîne sur TRAIN, sélectionne sur VALIDATION et ne consulte pas TEST."
)

try:
    import optuna
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except (ImportError, OSError) as exc:
    st.error("Impossible de charger PyTorch ou Optuna. Vérifie les dépendances.")
    st.exception(exc)
    st.stop()

optuna.logging.set_verbosity(optuna.logging.WARNING)


def clean_rates(raw: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage causal : pas d'interpolation des publications absentes."""
    data = raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    data["TIME_PERIOD"] = pd.to_datetime(data["TIME_PERIOD"], errors="coerce")
    data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    data = data.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    data = data.loc[np.isfinite(data["OBS_VALUE"]) & (data["OBS_VALUE"] > 0)]
    return (
        data.sort_values("TIME_PERIOD", kind="stable")
        .drop_duplicates("TIME_PERIOD", keep="last")
        .reset_index(drop=True)
    )


from src.dl_architectures import build_network
from src.dashboard_models import (
    load_checkpoint, evaluate_saved_checkpoint, session_to_checkpoint,
)


try:
    frame = clean_rates(get_data_or_stop())
except (KeyError, ValueError, TypeError) as exc:
    st.error(f"Impossible de préparer les données BCE : {exc}")
    st.stop()

values = frame["OBS_VALUE"].to_numpy(dtype=np.float64)
n = len(values)
if n < 250:
    st.error("Historique insuffisant pour les trois partitions.")
    st.stop()

train_end, val_end = int(n * 0.70), int(n * 0.85)
train_mean = float(values[:train_end].mean())
train_std = float(values[:train_end].std())
if train_std <= 0 or not np.isfinite(train_std):
    st.error("Écart-type de TRAIN invalide.")
    st.stop()

# Les statistiques de normalisation proviennent UNIQUEMENT de TRAIN.
scaled = ((values - train_mean) / train_std).astype(np.float32)


def make_sequences(lookback: int):
    """Cible au temps i ; fenêtre [i-lookback, i[ ; split selon la cible."""
    train_indices = np.arange(lookback, train_end)
    validation_indices = np.arange(train_end, val_end)

    def select(indices):
        X = np.stack([scaled[i - lookback:i] for i in indices])
        return X[..., None], scaled[indices]

    return *select(train_indices), *select(validation_indices)


# Référence naïve sur exactement les mêmes dates de validation pour tous les essais.
naive_mae = float(np.abs(
    values[train_end:val_end] - values[train_end - 1:val_end - 1]
).mean())

cols = st.columns(4)
cols[0].metric("TRAIN", f"{train_end:,}")
cols[1].metric("VALIDATION", f"{val_end-train_end:,}")
cols[2].metric("TEST réservé", f"{n-val_end:,}")
cols[3].metric("Naïf · MAE validation actuelle", f"{naive_mae:.6f}")

# Inventaire sauvegardé : visible dès l'ouverture, sans réentraînement.
st.subheader("Modèles déjà entraînés")
inventory = []
for candidate in ("LSTM", "GRU", "TCN"):
    existing_session = st.session_state.get(f"forex_optuna_{candidate}")
    if existing_session and existing_session.get("fingerprint") == sha256(values.tobytes()).hexdigest():
        inventory.append({
            "Architecture": candidate,
            "Disponibilité": "En mémoire de session",
            "MAE validation": f"{existing_session['mae']:.6f}",
        })
        continue
    stored_model, status = load_checkpoint(candidate)
    inventory.append({
        "Architecture": candidate,
        "Disponibilité": "Enregistré sur le PC" if stored_model else status,
        "MAE validation": (
            f"{stored_model['validation_mae']:.6f}"
            if stored_model and stored_model.get("validation_mae") is not None
            else "À retrouver" if stored_model else "—"
        ),
    })
st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
st.caption(
    "Les modèles enregistrés sont restaurés automatiquement. "
    "Cliquer sur « Lancer Optuna » sert uniquement à lancer un nouvel entraînement."
)

architecture = st.selectbox("Architecture", ["LSTM", "GRU", "TCN"])
a, b, c = st.columns(3)
n_trials = a.slider("Essais Optuna", 5, 30, 10)
max_epochs = b.slider("Époques max / essai", 10, 60, 30)
patience = c.slider("Early stopping · patience", 3, 12, 7)
st.caption(
    "Optuna explore : fenêtres 10/30/60, couches 1–2 (TCN jusqu'à 3), "
    "neurones 16/32/64, dropout, batchs et learning rate. "
    "TCN utilise des convolutions causales et des connexions résiduelles. "
    "Calcul CPU ; aucune cible TEST n'est lue."
)

fingerprint = sha256(values.tobytes()).hexdigest()
key = f"forex_optuna_{architecture}"

if st.button("▶ Lancer Optuna", type="primary"):
    artifacts = {}
    progress = st.progress(0.0)
    status = st.empty()

    def objective(trial):
        lookback = trial.suggest_categorical("lookback", [10, 30, 60])
        hidden = trial.suggest_categorical("hidden", [16, 32, 64])
        layers = trial.suggest_int("layers", 1, 3 if architecture == "TCN" else 2)
        dropout = (
            trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
            if layers > 1 else 0.0
        )
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])
        X_tr, y_tr, X_val, y_val = make_sequences(lookback)

        torch.manual_seed(42)
        model = build_network(architecture, hidden, layers, dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=batch, shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        x_val_tensor = torch.from_numpy(X_val)

        best_mae = float("inf")
        best_weights, best_epoch = None, 0
        no_improvement = 0
        history = []

        for epoch in range(1, max_epochs + 1):
            model.train()
            weighted_loss = 0.0
            for xx, yy in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xx), yy)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                weighted_loss += float(loss.item()) * len(xx)

            model.eval()
            with torch.no_grad():
                predictions = torch.cat([
                    model(chunk) for chunk in x_val_tensor.split(batch)
                ]).numpy()
            train_mse = weighted_loss / len(X_tr)
            validation_mse = float(np.mean((predictions - y_val) ** 2))
            validation_mae = float(np.mean(np.abs(predictions - y_val)) * train_std)
            if not np.isfinite(validation_mae):
                raise optuna.TrialPruned("Loss non finie")
            history.append({
                "Époque": epoch, "MSE TRAIN": train_mse,
                "MSE VALIDATION": validation_mse,
                "MAE VALIDATION": validation_mae,
            })
            if validation_mae < best_mae - 1e-7:
                best_mae, best_epoch = validation_mae, epoch
                best_weights = deepcopy({
                    name: param.detach().cpu().clone()
                    for name, param in model.state_dict().items()
                })
                no_improvement = 0
            else:
                no_improvement += 1

            trial.report(validation_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if no_improvement >= patience:
                break

        if best_weights is None:
            raise optuna.TrialPruned("Aucun modèle valide")
        trial.set_user_attr("best_epoch", best_epoch)
        artifacts[trial.number] = {
            "weights": best_weights, "history": history
        }
        return best_mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4, n_warmup_steps=5
        ),
    )

    def on_trial_complete(study, trial):
        progress.progress(len(study.trials) / n_trials)
        status.caption(f"Essai {len(study.trials)}/{n_trials} · {trial.state.name}")

    try:
        with st.spinner("Optimisation sur TRAIN et VALIDATION..."):
            study.optimize(
                objective, n_trials=n_trials,
                callbacks=[on_trial_complete], n_jobs=1,
            )
        best_trial = study.best_trial
    except (RuntimeError, ValueError) as exc:
        st.error(f"Optimisation interrompue : {exc}")
    else:
        st.session_state[key] = {
            "fingerprint": fingerprint,
            "params": best_trial.params,
            "mae": float(best_trial.value),
            "naive": naive_mae,
            "trials": study.trials_dataframe(),
            "artifacts": artifacts[best_trial.number],
            "mean": train_mean, "std": train_std,
            "train_end": str(frame.iloc[train_end-1]["TIME_PERIOD"].date()),
            "val_end": str(frame.iloc[val_end-1]["TIME_PERIOD"].date()),
        }
        # Sauvegarde locale automatique : réutilisable par le dashboard final.
        # Les poids restent dans artifacts/forex_models (hors dépôt Git).
        try:
            from src.dashboard_models import save_checkpoint
            file_saved = save_checkpoint(architecture, st.session_state[key])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            st.warning(
                "Optimisation terminée, mais sauvegarde locale impossible : "
                f"{error}. Utilise l'export .pt ci-dessous."
            )
        else:
            st.caption(f"Meilleurs poids sauvegardés automatiquement : {file_saved}")
        st.success("Optimisation terminée. Résultats conservés pour cette session.")

# À chaque réouverture : d'abord session courante, sinon checkpoint persistant.
saved = st.session_state.get(key)
saved_matches = saved is not None and saved.get("fingerprint") == fingerprint
if saved_matches:
    best_checkpoint = session_to_checkpoint(architecture, saved)
    saved_source = "Résultat de cette session"
else:
    best_checkpoint, saved_source = load_checkpoint(architecture)

if best_checkpoint is not None:
    st.subheader(f"Résultats enregistrés — {architecture}")
    st.success(
        f"Modèle {architecture} disponible : {saved_source}. "
        "Aucun nouvel entraînement nécessaire."
    )

    # Validation du modèle sur la tranche chronologique d'origine, pour éviter
    # qu'une publication BCE ajoutée demain ne change le score déjà enregistré.
    try:
        if best_checkpoint.get("validation_mae") is None or best_checkpoint.get("naive_mae") is None:
            with st.spinner("Récupération des métriques historiques sans entraînement…"):
                recovered = evaluate_saved_checkpoint(best_checkpoint, architecture, frame)
            metric_mae = float(recovered["mae"])
            metric_naive = float(recovered["naive_mae"])
            historical_dates = (
                recovered["train_end"], recovered["validation_end"]
            )
            st.caption(
                "Ce fichier .pt est d'une version précédente : MAE recalculée "
                "sur sa validation d'origine, sans optimiser ni réentraîner."
            )
        else:
            metric_mae = float(best_checkpoint["validation_mae"])
            metric_naive = float(best_checkpoint["naive_mae"])
            historical_dates = (
                best_checkpoint["train_end"], best_checkpoint["validation_end"]
            )
    except (KeyError, ValueError, RuntimeError, TypeError, IndexError) as exc:
        st.error(f"Impossible de reconstituer les métriques historiques : {exc}")
        metric_mae = None
        metric_naive = None
        historical_dates = None

    if metric_mae is not None and metric_naive is not None:
        c1, c2 = st.columns(2)
        c1.metric("MAE validation du modèle entraîné", f"{metric_mae:.6f}")
        c2.metric("MAE naïf · même validation", f"{metric_naive:.6f}")
        if metric_mae < metric_naive:
            st.success("Le réseau dépasse le naïf sur la validation d'origine.")
        else:
            st.info("Le naïf reste meilleur sur la validation d'origine.")

    if historical_dates:
        st.caption(
            "Entraînement jusqu'au " + historical_dates[0] +
            " · Validation jusqu'au " + historical_dates[1] +
            " (dates figées lors de l'optimisation)."
        )
    st.write("**Hyperparamètres du modèle sauvegardé**")
    st.json(best_checkpoint["parameters"])

    # Les anciens checkpoints n'enregistraient pas les essais ni la courbe ;
    # les fichiers récents conservent ces métadonnées JSON simples.
    trials_json = best_checkpoint.get("trials_json", "")
    if saved_matches and hasattr(saved.get("trials"), "to_csv"):
        trials_frame = saved["trials"]
    elif trials_json:
        try:
            from io import StringIO
            trials_frame = pd.read_json(StringIO(trials_json), orient="records")
        except (ValueError, TypeError):
            trials_frame = None
    else:
        trials_frame = None
    if trials_frame is not None and not trials_frame.empty:
        with st.expander("Historique des essais Optuna", expanded=False):
            st.dataframe(trials_frame, width="stretch")
        st.download_button(
            "Exporter les essais (CSV)",
            trials_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"eurusd_optuna_{architecture.lower()}_essais.csv",
            mime="text/csv",
        )

    history = best_checkpoint.get("training_history") or []
    if history:
        history_frame = pd.DataFrame(history)
        st.line_chart(
            history_frame.set_index("Époque")[["MSE TRAIN", "MSE VALIDATION"]],
            width="stretch",
        )
    else:
        st.caption(
            "Courbe d'entraînement indisponible dans cet ancien checkpoint ; "
            "les poids et les hyperparamètres sont bien conservés."
        )

    checkpoint_data = BytesIO()
    torch.save(best_checkpoint, checkpoint_data)
    st.download_button(
        "Exporter le modèle sauvegardé (.pt)", checkpoint_data.getvalue(),
        file_name=f"eurusd_optuna_{architecture.lower()}_best.pt",
        mime="application/octet-stream",
    )
else:
    st.info(
        f"Aucun modèle {architecture} sauvegardé sur ce PC. "
        "Si tu as encore son .pt exporté, conserve-le ; sinon "
        "Optuna doit être lancé une seule fois pour créer un checkpoint."
    )

st.warning(
    "LSTM, GRU et TCN sont entraînés et optimisés sur les mêmes partitions. "
    "Leurs prévisions sont comparables dans la phase 9 bis (validation), "
    "et ne sont pas injectées dans le test de la phase 9. Après sélection sur validation, "
    "une nouvelle période hors échantillon doit confirmer la généralisation."
)
