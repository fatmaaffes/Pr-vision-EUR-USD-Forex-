"""Optuna sur LSTM/GRU EUR/USD : validation chronologique, sans exploiter TEST."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from src.bce import get_data_or_stop

st.title("🔬 Phase 8 bis — Optimisation automatique LSTM / GRU")
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


class RecurrentNet(nn.Module):
    def __init__(self, architecture: str, hidden: int, layers: int, dropout: float):
        super().__init__()
        klass = nn.LSTM if architecture == "LSTM" else nn.GRU
        self.recurrent = klass(
            input_size=1, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0, batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, X):
        sequence, _ = self.recurrent(X)
        return self.head(sequence[:, -1, :]).squeeze(-1)


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
cols[3].metric("Naïf · MAE validation", f"{naive_mae:.6f}")

architecture = st.selectbox("Architecture", ["LSTM", "GRU"])
a, b, c = st.columns(3)
n_trials = a.slider("Essais Optuna", 5, 30, 10)
max_epochs = b.slider("Époques max / essai", 10, 60, 30)
patience = c.slider("Early stopping · patience", 3, 12, 7)
st.caption(
    "Optuna explore : fenêtres 10/30/60, couches 1–2, neurones 16/32/64, "
    "dropout, taille des batchs et learning rate. Calcul CPU."
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
        layers = trial.suggest_int("layers", 1, 2)
        dropout = (
            trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
            if layers > 1 else 0.0
        )
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128])
        X_tr, y_tr, X_val, y_val = make_sequences(lookback)

        torch.manual_seed(42)
        model = RecurrentNet(architecture, hidden, layers, dropout)
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

saved = st.session_state.get(key)
if saved is not None and saved["fingerprint"] == fingerprint:
    st.subheader(f"Résultats — {architecture}")
    c1, c2 = st.columns(2)
    c1.metric("Meilleure MAE validation", f"{saved['mae']:.6f}")
    c2.metric("Naïf · MAE validation", f"{saved['naive']:.6f}")
    st.json(saved["params"])
    st.dataframe(saved["trials"], width="stretch")
    st.line_chart(
        pd.DataFrame(saved["artifacts"]["history"])
        .set_index("Époque")[["MSE TRAIN", "MSE VALIDATION"]],
        width="stretch",
    )
    st.download_button(
        "Exporter les essais (CSV)",
        saved["trials"].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"eurusd_optuna_{architecture.lower()}_essais.csv",
        mime="text/csv",
    )

    checkpoint = BytesIO()
    torch.save({
        "architecture": architecture,
        "parameters": saved["params"],
        "model_state_dict": saved["artifacts"]["weights"],
        "standardization_mean": saved["mean"],
        "standardization_std": saved["std"],
        "train_end": saved["train_end"],
        "validation_end": saved["val_end"],
    }, checkpoint)
    st.download_button(
        "Exporter le meilleur modèle (.pt)", checkpoint.getvalue(),
        file_name=f"eurusd_optuna_{architecture.lower()}_best.pt",
        mime="application/octet-stream",
    )

    if saved["mae"] < saved["naive"]:
        st.success("Le réseau dépasse la baseline naïve sur la validation.")
    else:
        st.info("La baseline naïve reste meilleure sur la validation.")

st.warning(
    "Cette expérience univariée est indépendante des phases 8 et 9 actuelles : "
    "elle n'injecte pas ses prévisions dans la phase 9. "
    "Après sélection répétée sur validation, une nouvelle évaluation "
    "hors échantillon doit confirmer la généralisation."
)
