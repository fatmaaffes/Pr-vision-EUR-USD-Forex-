"""Phase 9 bis : comparaison de Naïf, LSTM, GRU et TCN par volatilité."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.bce import get_data_or_stop
from src.volatility_regimes import clean_rates, validation_frame, compute_metrics
from src.dashboard_models import load_checkpoint

st.title("🌡️ Phase 9 bis — Comparaison par volatilité")
st.write(
    "Quel modèle prédit le mieux EUR/USD lorsque la volatilité récente est élevée ? "
    "On compare les prévisions **sur les mêmes dates de VALIDATION**, "
    "sans réentraîner les réseaux et sans utiliser les cibles de TEST."
)

try:
    import torch
    from torch import nn
except (ImportError, OSError) as exc:
    st.error("PyTorch n'est pas disponible dans cet environnement.")
    st.exception(exc)
    st.stop()


from src.dl_architectures import build_network


frame = clean_rates(get_data_or_stop())
values = frame["OBS_VALUE"].to_numpy(dtype=np.float64)
if len(values) < 250:
    st.warning("Historique BCE insuffisant.")
    st.stop()

train_end = int(len(values) * 0.70)
val_end = int(len(values) * 0.85)
last_train_date = str(frame.iloc[train_end - 1]["TIME_PERIOD"].date())
last_val_date = str(frame.iloc[val_end - 1]["TIME_PERIOD"].date())
mean = float(values[:train_end].mean())
std = float(values[:train_end].std())
fingerprint = sha256(values.tobytes()).hexdigest()

table, threshold = validation_frame(
    frame, train_end=train_end, val_end=val_end, window=20
)

st.subheader("1. Identifier les régimes de marché")
a, b, c = st.columns(3)
a.metric("Seuil de volatilité (TRAIN)", f"{threshold:.4f} %")
b.metric("Validation · faible", int((table["Régime"] == "Faible volatilité").sum()))
c.metric("Validation · forte", int((table["Régime"] == "Forte volatilité").sum()))
st.caption(
    "Seuil : médiane de la volatilité glissante de 20 rendements, calculée "
    "uniquement sur TRAIN. Pour prédire la date i, la volatilité utilise "
    "les rendements jusqu'à i-1 (aucune information future)."
)

st.subheader("2. Modèles optimisés disponibles")
st.write(
    "LSTM, GRU et TCN sont récupérés automatiquement depuis la session "
    "Streamlit ou les modèles enregistrés sur ce PC. L'import manuel "
    "n'est utile qu'en cas de sauvegarde absente."
)


def get_checkpoint(architecture: str):
    """Sélectionner un modèle de session valide ou charger son .pt exporté."""
    saved = st.session_state.get(f"forex_optuna_{architecture}")
    if saved and saved.get("fingerprint") == fingerprint:
        st.success(f"{architecture} : poids retrouvés en mémoire de session.")
        return {
            "params": saved["params"],
            "weights": saved["artifacts"]["weights"],
            "mean": float(saved["mean"]),
            "std": float(saved["std"]),
            "train_end": saved["train_end"],
            "val_end": saved["val_end"],
        }

    if saved:
        st.warning(
            f"{architecture} : les données BCE ont changé depuis l'optimisation. "
            "Les anciens résultats de session ne sont pas réutilisés."
        )

    checkpoint_local, local_status = load_checkpoint(architecture)
    if checkpoint_local is not None:
        st.success(f"{architecture} : {local_status}.")
        return {
            "params": checkpoint_local["parameters"],
            "weights": checkpoint_local["model_state_dict"],
            "mean": float(checkpoint_local["standardization_mean"]),
            "std": float(checkpoint_local["standardization_std"]),
            "train_end": checkpoint_local["train_end"],
            "val_end": checkpoint_local["validation_end"],
        }

    uploaded = st.file_uploader(
        f"Importer les poids {architecture} depuis Optuna (.pt)",
        type=["pt", "pth"],
        key=f"regime_upload_{architecture}",
    )
    if uploaded is None:
        st.caption(f"{architecture} : aucun modèle disponible pour le moment.")
        return None

    try:
        ckpt = torch.load(
            BytesIO(uploaded.getvalue()), map_location="cpu", weights_only=True
        )
        if ckpt["architecture"] != architecture:
            raise ValueError("Architecture du fichier incorrecte.")
        return {
            "params": ckpt["parameters"],
            "weights": ckpt["model_state_dict"],
            "mean": float(ckpt["standardization_mean"]),
            "std": float(ckpt["standardization_std"]),
            "train_end": ckpt["train_end"],
            "val_end": ckpt["validation_end"],
        }
    except (KeyError, ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Fichier {architecture} incompatible : {exc}")
        return None


checkpoints = {
    arch: get_checkpoint(arch)
    for arch in ("LSTM", "GRU", "TCN")
}


def predict_validation(arch: str, checkpoint: dict) -> np.ndarray:
    """Inférence sur validation, sans entraîner et sans lire les cibles test."""
    if (
        checkpoint["train_end"] != last_train_date
        or checkpoint["val_end"] != last_val_date
    ):
        raise ValueError(
            f"{arch} : les dates TRAIN/VALIDATION ne correspondent pas "
            "au modèle optimisé. Ne mélange pas deux historiques BCE."
        )
    if not (
        np.isclose(checkpoint["mean"], mean, rtol=0, atol=1e-6)
        and np.isclose(checkpoint["std"], std, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            f"{arch} : le scaler du checkpoint ne correspond pas à TRAIN."
        )

    params = checkpoint["params"]
    lookback = int(params["lookback"])
    if not (1 <= lookback < train_end):
        raise ValueError(f"{arch} : longueur de fenêtre invalide.")

    network = build_network(
        arch, int(params["hidden"]), int(params["layers"]), float(params.get("dropout", 0.0))
    )
    network.load_state_dict(checkpoint["weights"], strict=True)
    network.eval()

    # Toujours la normalisation de TRAIN, comme pendant Optuna.
    scaled = ((values - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    idx = np.arange(train_end, val_end)
    X = np.stack([scaled[i - lookback:i] for i in idx])
    X = torch.from_numpy(X[:, :, None])

    with torch.no_grad():
        raw_outputs = np.concatenate([
            network(batch).cpu().numpy()
            for batch in X.split(256)
        ])
    return raw_outputs.astype(np.float64) * checkpoint["std"] + checkpoint["mean"]


columns = ["Prévision — Naïf"]
for arch in ("LSTM", "GRU"):
    checkpoint = checkpoints[arch]
    if checkpoint is None:
        continue
    try:
        predictions = predict_validation(arch, checkpoint)
        dated = pd.Series(
            predictions,
            index=pd.DatetimeIndex(
                frame.iloc[train_end:val_end]["TIME_PERIOD"].to_numpy()
            ),
        )
        col_name = f"Prévision — {arch} Optuna"
        table[col_name] = pd.DatetimeIndex(table["Date cible"]).map(dated)
        if table[col_name].isna().any():
            raise ValueError("Certaines prévisions ne sont pas alignées sur les dates.")
        columns.append(col_name)
    except (KeyError, ValueError, RuntimeError, TypeError) as exc:
        st.error(f"{arch} : impossible de produire les prévisions : {exc}")

st.subheader("3. Comparer les erreurs selon la volatilité")
metrics = compute_metrics(table, columns)
st.dataframe(metrics.round(5), hide_index=True, width="stretch")
st.caption(
    "MAE et RMSE en USD par EUR. Gain MAE vs naïf > 0 signifie que "
    "le modèle fait moins d'erreurs que le naïf dans ce régime. "
    "La direction du naïf n'est pas applicable : il prédit une variation nulle."
)

if len(columns) > 1:
    st.write("### MAE par modèle et par régime (plus faible = meilleure)")
    chart_data = metrics[["Régime", "Modèle", "MAE"]].copy()
    grouped_chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Modèle:N", title="Modèle", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Régime:N"),
            y=alt.Y("MAE:Q", title="MAE (USD/EUR)", scale=alt.Scale(zero=True)),
            color=alt.Color("Régime:N", title="Régime de volatilité"),
            tooltip=[
                alt.Tooltip("Modèle:N"),
                alt.Tooltip("Régime:N"),
                alt.Tooltip("MAE:Q", format=".6f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(grouped_chart, width="stretch")
    high = metrics[metrics["Régime"] == "Forte volatilité"]
    winner = high.loc[high["MAE"].idxmin()]
    st.info(
        f"Sur la validation à forte volatilité, la plus faible MAE observée "
        f"est celle du modèle **{winner['Modèle']}** : "
        f"{winner['MAE']:.6f}. Ce classement ne garantit pas "
        "les performances futures."
    )
else:
    st.warning(
        "Seul le naïf est disponible. Pour comparer LSTM/GRU/TCN, "
        "ouvre Phase 8 bis, lance les optimisations ; les meilleurs poids "
        "seront enregistrés automatiquement pour cette page."
    )

st.subheader("4. Examiner les dates de forte volatilité")
high_days = table.loc[table["Régime"] == "Forte volatilité"].copy()
for col in columns:
    high_days[f"Erreur absolue — {col.removeprefix('Prévision — ')}"] = (
        high_days[col] - high_days["EUR/USD réel"]
    ).abs()
st.dataframe(high_days.tail(30).round(6), hide_index=True, width="stretch")

st.download_button(
    "📥 Exporter la comparaison complète (VALIDATION CSV)",
    data=table.to_csv(index=False).encode("utf-8-sig"),
    file_name="eurusd_validation_regimes_volatilite.csv",
    mime="text/csv",
)

st.warning(
    "Analyse exploratoire effectuée uniquement sur VALIDATION, qui a déjà "
    "servi à sélectionner les hyperparamètres avec Optuna. Ces résultats "
    "ne constituent PAS une validation indépendante. "
    "Le TEST historique ayant déjà été consulté, une période future "
    "réservée sera nécessaire pour confirmer un éventuel avantage."
)
