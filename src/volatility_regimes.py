"""Outils testables de comparaison EUR/USD par régimes de volatilité.

Chaque régime est déterminé uniquement par les rendements connus avant
la date cible. Le seuil est estimé uniquement sur TRAIN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_rates(raw: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage identique à la page Optuna (sans interpolation)."""
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


def past_volatility(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Volatilité au temps cible i basée sur les rendements i-20..i-1."""
    if window < 2:
        raise ValueError("La fenêtre de volatilité doit être >= 2.")
    series = pd.Series(np.asarray(values, dtype=float))
    returns = series.pct_change(fill_method=None) * 100.0
    return (
        returns.shift(1)
        .rolling(window=window, min_periods=window)
        .std()
        .to_numpy(dtype=float)
    )


def validation_frame(
    rates: pd.DataFrame,
    train_end: int,
    val_end: int,
    window: int = 20,
) -> tuple[pd.DataFrame, float]:
    """Données de validation à mêmes dates ; seuil médian de TRAIN."""
    values = rates["OBS_VALUE"].to_numpy(dtype=float)
    if not (window + 2 < train_end < val_end <= len(values)):
        raise ValueError("Partitions chronologiques incompatibles.")

    vol = past_volatility(values, window=window)
    eligible_train = vol[:train_end]
    eligible_train = eligible_train[np.isfinite(eligible_train)]
    if not len(eligible_train):
        raise ValueError("TRAIN n'a pas de volatilité glissante valide.")
    threshold = float(np.median(eligible_train))

    idx = np.arange(train_end, val_end)
    table = pd.DataFrame({
        "Date cible": rates.iloc[idx]["TIME_PERIOD"].to_numpy(),
        "EUR/USD réel": values[idx],
        "Dernier EUR/USD connu": values[idx - 1],
        "Volatilité passée (%)": vol[idx],
        "Prévision — Naïf": values[idx - 1],
    })
    table = table.loc[np.isfinite(table["Volatilité passée (%)"])].reset_index(drop=True)
    table["Régime"] = np.where(
        table["Volatilité passée (%)"] >= threshold,
        "Forte volatilité",
        "Faible volatilité",
    )
    return table, threshold


def compute_metrics(
    table: pd.DataFrame,
    prediction_columns: list[str],
) -> pd.DataFrame:
    """MAE/RMSE à dates strictement identiques, par régime."""
    rows = []
    for regime in ("Faible volatilité", "Forte volatilité"):
        segment = table.loc[table["Régime"] == regime]
        if segment.empty:
            continue
        truth = segment["EUR/USD réel"].to_numpy(dtype=float)
        last = segment["Dernier EUR/USD connu"].to_numpy(dtype=float)
        naive_mae = float(np.mean(np.abs(truth - last)))
        for column in prediction_columns:
            predicted = segment[column].to_numpy(dtype=float)
            error = predicted - truth
            if not np.isfinite(predicted).all():
                raise ValueError(f"Prévision non finie : {column}")
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            valid_direction = np.sign(truth - last) != 0
            if column == "Prévision — Naïf" or not valid_direction.any():
                direction = np.nan
            else:
                direction = 100 * float(np.mean(
                    np.sign(predicted[valid_direction] - last[valid_direction])
                    == np.sign(truth[valid_direction] - last[valid_direction])
                ))
            improvement = (
                100.0 * (naive_mae - mae) / naive_mae
                if naive_mae > 0 else np.nan
            )
            rows.append({
                "Régime": regime,
                "Modèle": column.removeprefix("Prévision — "),
                "Observations": len(segment),
                "MAE": mae,
                "RMSE": rmse,
                "Direction (%)": direction,
                "Gain MAE vs naïf (%)": improvement,
            })
    return pd.DataFrame(rows)
