"""Tests de restauration d'un entraînement Optuna depuis les poids du disque."""
import numpy as np
import pandas as pd
import pytest
import torch

import src.dashboard_models as dm
from src.dl_architectures import build_network


def example_training(architecture="GRU"):
    indices = np.arange(220, dtype=float)
    values = 1.10 + 0.0002 * indices + 0.005 * np.sin(indices / 8)
    dates = pd.bdate_range("2025-01-01", periods=len(values))
    frame = pd.DataFrame({"TIME_PERIOD": dates, "OBS_VALUE": values})
    train_end, val_end = 140, 190
    params = {"lookback": 10, "hidden": 8, "layers": 1, "batch_size": 32,
              "learning_rate": 0.001}
    net = build_network(architecture, hidden=8, layers=1, dropout=0.0)
    saved = {
        "params": params,
        "mae": 0.003651,
        "naive": 0.003649,
        "fingerprint": "fingerprint-placeholder",
        "mean": float(values[:train_end].mean()),
        "std": float(values[:train_end].std()),
        "train_end": str(dates[train_end - 1].date()),
        "val_end": str(dates[val_end - 1].date()),
        "trials": pd.DataFrame([{"number": 0, "value": 0.003651}]),
        "artifacts": {
            "weights": net.state_dict(),
            "history": [
                {"Époque": 1, "MSE TRAIN": 0.01,
                 "MSE VALIDATION": 0.011, "MAE VALIDATION": 0.003651}
            ],
        },
    }
    return frame, saved


@pytest.mark.parametrize("architecture", ["LSTM", "GRU", "TCN"])
def test_checkpoint_reload_without_retraining(tmp_path, monkeypatch, architecture):
    monkeypatch.setattr(dm, "MODEL_DIR", tmp_path)
    frame, saved = example_training(architecture)
    path = dm.save_checkpoint(architecture, saved)
    assert path.is_file()

    loaded, source = dm.load_checkpoint(architecture)
    assert loaded is not None
    assert source == "Modèle enregistré sur le PC"
    assert loaded["validation_mae"] == saved["mae"]
    assert loaded["naive_mae"] == saved["naive"]
    assert loaded["training_history"] == saved["artifacts"]["history"]
    assert loaded["trials_json"]
    assert loaded["parameters"] == saved["params"]

    result = dm.evaluate_saved_checkpoint(loaded, architecture, frame)
    assert result["validation_count"] == 50
    assert np.isfinite(result["mae"])
    assert np.isfinite(result["naive_mae"])


def test_checkpoint_old_format_and_appended_bce_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "MODEL_DIR", tmp_path)
    frame, saved = example_training("TCN")
    original = dm.session_to_checkpoint("TCN", saved)
    # Format historique du .pt : poids et scaler, sans scores ni historique.
    old = {k: original[k] for k in (
        "architecture", "parameters", "model_state_dict",
        "standardization_mean", "standardization_std", "train_end",
        "validation_end"
    )}
    path = dm.checkpoint_path("TCN")
    torch.save(old, path)
    loaded, _ = dm.load_checkpoint("TCN")
    before = dm.evaluate_saved_checkpoint(loaded, "TCN", frame)

    # La nouvelle publication BCE ne change pas la validation historique.
    updated = pd.concat(
        [frame, pd.DataFrame([{
            "TIME_PERIOD": frame["TIME_PERIOD"].iloc[-1] + pd.Timedelta(days=1),
            "OBS_VALUE": 1.30
        }])],
        ignore_index=True,
    )
    after = dm.evaluate_saved_checkpoint(loaded, "TCN", updated)
    assert before == after


def test_training_data_change_prevents_invalid_reconstruction():
    frame, saved = example_training("LSTM")
    checkpoint = dm.session_to_checkpoint("LSTM", saved)
    corrupted = frame.copy()
    corrupted.loc[10, "OBS_VALUE"] += 0.6
    with pytest.raises(ValueError, match="TRAIN"):
        dm.evaluate_saved_checkpoint(checkpoint, "LSTM", corrupted)
