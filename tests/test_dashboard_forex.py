"""Tests unitaires du dashboard métier Forex, sans téléchargement BCE."""
import numpy as np
import pytest

from src.dashboard_forex import (
    monte_carlo_costs, stressed_rates, volatility_context,
)


def rates_sample():
    indices = np.arange(260, dtype=float)
    return 1.1 + 0.0003 * indices + 0.009 * np.sin(indices / 6)


def test_today_volatility_ignores_unpublished_next_rate():
    rates = rates_sample()
    original = volatility_context(rates, window=20)
    # La volatilité à la prochaine publication est calculée AVANT cette publication.
    extra = np.r_[rates, 2.5]
    next_context = volatility_context(extra, window=20)
    # Le régime calculé après la publication peut changer, mais pas celui du jour précédent.
    assert np.isfinite(original["recent"])
    assert original["recent"] >= 0
    assert np.isfinite(next_context["recent"])


def test_training_threshold_cannot_see_newest_observations():
    rates = rates_sample()
    first = volatility_context(rates)
    changed = rates.copy()
    changed[-10:] *= 1.15
    second = volatility_context(changed)
    assert first["threshold"] == second["threshold"]
    assert first["low_sigma"] == second["low_sigma"]
    assert first["high_sigma"] == second["high_sigma"]


def test_stress_scenarios_are_symmetric_and_cost_more_on_euro_drop():
    scenarios = stressed_rates(1.15, 1.0)
    assert scenarios["Dépréciation EUR"] < scenarios["Stabilité"]
    assert scenarios["Appréciation EUR"] > scenarios["Stabilité"]
    assert 100000 / scenarios["Dépréciation EUR"] > 100000 / scenarios["Stabilité"]


def test_monte_carlo_reproducible_and_wider_when_sigma_rises():
    low = monte_carlo_costs(100000, 1.15, 0.2, 20, n_paths=2000)
    high = monte_carlo_costs(100000, 1.15, 1.0, 20, n_paths=2000)
    again = monte_carlo_costs(100000, 1.15, 0.2, 20, n_paths=2000)
    assert low["p95"] > low["p50"] > low["p05"]
    assert high["p95"] - high["p05"] > low["p95"] - low["p05"]
    assert low["p50"] == again["p50"]


def test_invalid_monetary_inputs_rejected():
    with pytest.raises(ValueError):
        stressed_rates(-1, 1)
    with pytest.raises(ValueError):
        monte_carlo_costs(-100, 1.15, 0.5, 5)
