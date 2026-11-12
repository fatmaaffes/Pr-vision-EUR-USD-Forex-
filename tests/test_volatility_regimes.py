"""Tests de non-fuite temporelle et des métriques par régime."""
import numpy as np
import pandas as pd

from src.volatility_regimes import (
    past_volatility, clean_rates, validation_frame, compute_metrics,
)


def sample_rates():
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    # Des rendements oscillants, strictement positifs en niveau.
    rates = 1.10 + 0.0009 * np.arange(120) + 0.015 * np.sin(np.arange(120) / 3.0)
    return pd.DataFrame({"TIME_PERIOD": idx, "OBS_VALUE": rates})


def test_past_volatility_excludes_current_and_future():
    values = sample_rates()["OBS_VALUE"].to_numpy()
    base = past_volatility(values, window=5)
    updated = values.copy()
    updated[70:] += 0.30
    changed = past_volatility(updated, window=5)
    np.testing.assert_allclose(base[:71], changed[:71], equal_nan=True)


def test_train_threshold_unaffected_by_validation_values():
    rates = sample_rates()
    cleaned = clean_rates(rates)
    original, original_threshold = validation_frame(cleaned, 80, 100)
    updated = cleaned.copy()
    updated.loc[80:, "OBS_VALUE"] *= 1.5
    _, next_threshold = validation_frame(updated, 80, 100)
    assert original_threshold == next_threshold
    assert len(original) == 20


def test_identical_prediction_dates_and_metrics():
    cleaned = clean_rates(sample_rates())
    val, _ = validation_frame(cleaned, 80, 100)
    val["Prévision — parfait"] = val["EUR/USD réel"]
    result = compute_metrics(val, ["Prévision — Naïf", "Prévision — parfait"])
    assert set(result["Modèle"]) == {"Naïf", "parfait"}
    assert all(result[result["Modèle"] == "parfait"]["MAE"] == 0)
    assert all(result["Observations"] > 0)
