"""Sauvegarde locale et inférence des réseaux Optuna sur la prochaine BCE.

Les poids sont persistés dans artifacts/forex_models (ignorés par Git).
Aucun réentraînement automatique ni utilisation de la cible future.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "forex_models"


def checkpoint_path(architecture: str) -> Path:
    if architecture not in ("LSTM", "GRU"):
        raise ValueError("Architecture inconnue.")
    return MODEL_DIR / f"{architecture.lower()}_optuna_best.pt"


def session_to_checkpoint(architecture: str, saved: dict) -> dict:
    """Convertir l'objet session de la Phase 8 bis en checkpoint torch."""
    return {
        "architecture": architecture,
        "parameters": saved["params"],
        "model_state_dict": saved["artifacts"]["weights"],
        "standardization_mean": float(saved["mean"]),
        "standardization_std": float(saved["std"]),
        "train_end": str(saved["train_end"]),
        "validation_end": str(saved["val_end"]),
    }


def save_checkpoint(architecture: str, saved: dict) -> Path:
    """Persister le meilleur modèle après Optuna (sur le PC qui exécute Streamlit)."""
    import torch

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_path(architecture)
    torch.save(session_to_checkpoint(architecture, saved), destination)
    return destination


def load_checkpoint(architecture: str, session: dict | None = None) -> tuple[dict | None, str]:
    """Préférer les poids de la session puis le fichier local, sans upload."""
    if session:
        try:
            return session_to_checkpoint(architecture, session), "Session Streamlit"
        except (KeyError, TypeError):
            pass
    path = checkpoint_path(architecture)
    if not path.is_file():
        return None, "Aucun modèle entraîné disponible"
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=True), "Modèle enregistré sur le PC"
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return None, f"Impossible de charger le modèle : {exc}"


def predict_next(checkpoint: dict, architecture: str, rates: pd.DataFrame) -> float:
    """Prévoir J+1 à partir des derniers taux connus et du scaler TRAIN original."""
    import torch
    from torch import nn

    if checkpoint.get("architecture") != architecture:
        raise ValueError("Architecture incompatible avec le modèle sauvegardé.")

    dates = pd.to_datetime(rates["TIME_PERIOD"])
    train_cut = pd.Timestamp(checkpoint["train_end"])
    val_cut = pd.Timestamp(checkpoint["validation_end"])
    train_positions = np.flatnonzero(dates.to_numpy() == train_cut.to_datetime64())
    val_positions = np.flatnonzero(dates.to_numpy() == val_cut.to_datetime64())
    if len(train_positions) != 1 or len(val_positions) != 1:
        raise ValueError("Les dates d'entraînement d'origine ne sont plus présentes.")
    train_count = int(train_positions[0]) + 1
    if int(val_positions[0]) <= train_positions[0]:
        raise ValueError("Dates d'entraînement et de validation incohérentes.")

    values = rates["OBS_VALUE"].to_numpy(dtype=np.float64)
    scaler_mean = float(checkpoint["standardization_mean"])
    scaler_std = float(checkpoint["standardization_std"])
    if not np.isfinite(scaler_std) or scaler_std <= 0:
        raise ValueError("Normalisation enregistrée invalide.")
    if not (
        np.isclose(values[:train_count].mean(), scaler_mean, rtol=0, atol=1e-6)
        and np.isclose(values[:train_count].std(), scaler_std, rtol=0, atol=1e-6)
    ):
        raise ValueError("Les données TRAIN ont changé : réoptimiser le modèle.")

    config = checkpoint["parameters"]
    hidden = int(config["hidden"])
    layers = int(config["layers"])
    lookback = int(config["lookback"])
    if lookback < 2 or lookback > len(values) or hidden < 1 or layers < 1:
        raise ValueError("Hyperparamètres du réseau invalides.")
    dropout = float(config.get("dropout", 0.0))
    network_cls = nn.LSTM if architecture == "LSTM" else nn.GRU

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = network_cls(
                input_size=1, hidden_size=hidden, num_layers=layers,
                dropout=dropout if layers > 1 else 0.0, batch_first=True,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            output, _ = self.recurrent(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    network = Network()
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.eval()
    features = ((values[-lookback:] - scaler_mean) / scaler_std).astype(np.float32)
    with torch.no_grad():
        pred_scaled = float(
            network(torch.from_numpy(features.reshape(1, lookback, 1))).item()
        )
    forecast = pred_scaled * scaler_std + scaler_mean
    if not np.isfinite(forecast) or forecast <= 0:
        raise ValueError("Le modèle a produit un taux invalide.")
    return float(forecast)
