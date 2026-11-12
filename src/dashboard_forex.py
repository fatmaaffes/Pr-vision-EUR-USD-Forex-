"""Calculs métier du dashboard Forex (sans Streamlit ni dépendance torch).

Les scénarios sont hypothétiques ; ils ne constituent pas des prévisions conditionnelles
du GRU. Le seuil et les statistiques de référence sont estimés sur TRAIN.
"""
from __future__ import annotations

import numpy as np
from src.volatility_regimes import past_volatility


def volatility_context(values, train_fraction=0.70, window=20):
    """Volatilité des 20 derniers rendements publiés et régimes de TRAIN."""
    rates = np.asarray(values, dtype=float)
    if len(rates) < max(100, window + 3) or not np.all(np.isfinite(rates)):
        raise ValueError("Historique BCE insuffisant ou non valide.")
    if np.any(rates <= 0):
        raise ValueError("Les taux EUR/USD doivent être positifs.")

    # La position n est celle de la future publication, encore inconnue.
    # La volatilité qui lui est associée utilise les rendements jusqu'à n-1.
    vol = past_volatility(rates, window=window)
    future_vol = past_volatility(
        np.concatenate([rates, np.array([np.nan])]), window=window
    )[-1]
    boundary = int(len(rates) * train_fraction)
    historical = vol[:boundary]
    historical = historical[np.isfinite(historical)]
    if not historical.size or not np.isfinite(future_vol):
        raise ValueError("Impossible de calculer la volatilité historique.")

    threshold = float(np.median(historical))
    low = historical[historical < threshold]
    high = historical[historical >= threshold]
    low_sigma = float(np.median(low)) if low.size else threshold
    high_sigma = float(np.median(high)) if high.size else threshold
    return {
        "recent": float(future_vol),
        "threshold": threshold,
        "regime": "Forte" if future_vol >= threshold else "Faible",
        "low_sigma": low_sigma,
        "high_sigma": high_sigma,
        "last_rate": float(rates[-1]),
    }


def stressed_rates(anchor: float, sigma_percent: float):
    """Scénarios illustratifs ± 1 écart-type sur un jour de publication."""
    if anchor <= 0 or sigma_percent < 0:
        raise ValueError("Taux ou volatilité invalide.")
    shift = sigma_percent / 100.0
    return {
        "Dépréciation EUR": anchor * (1 - shift),
        "Stabilité": anchor,
        "Appréciation EUR": anchor * (1 + shift),
    }


def monte_carlo_costs(
    usd_amount: float,
    anchor: float,
    sigma_percent: float,
    horizon: int,
    n_paths: int = 5000,
    seed: int = 42,
):
    """Coût EUR d'une facture USD selon un GBM simplifié sans drift estimé.

    L'amplitude de risque varie avec sigma et sqrt(horizon).
    Ce calcul ne déduit pas d'information sur les événements politiques.
    """
    if usd_amount <= 0 or anchor <= 0 or sigma_percent < 0:
        raise ValueError("Montant, taux ou volatilité invalide.")
    if not 1 <= horizon <= 260 or n_paths < 100:
        raise ValueError("Horizon ou nombre de simulations invalide.")
    sigma = sigma_percent / 100.0
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)
    terminal = anchor * np.exp(
        -0.5 * sigma * sigma * horizon
        + sigma * np.sqrt(horizon) * shocks
    )
    costs = usd_amount / terminal
    return {
        "p05": float(np.quantile(costs, 0.05)),
        "p50": float(np.quantile(costs, 0.50)),
        "p95": float(np.quantile(costs, 0.95)),
        "mean": float(np.mean(costs)),
        "today": float(usd_amount / anchor),
        "costs": costs,
    }
