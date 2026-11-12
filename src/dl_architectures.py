"""Architectures Deep Learning du projet Forex.

La factory conserve exactement les noms des modules LSTM/GRU déjà entraînés
(recurrent et head), afin de relire leurs checkpoints historiques.
Les convolutions TCN utilisent uniquement les observations passées.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentNet(nn.Module):
    """Prévision J+1 à partir de la dernière sortie d'un LSTM ou GRU."""

    def __init__(self, architecture: str, hidden: int, layers: int, dropout: float):
        super().__init__()
        if architecture not in ("LSTM", "GRU"):
            raise ValueError(f"Réseau récurrent non supporté : {architecture}")
        klass = nn.LSTM if architecture == "LSTM" else nn.GRU
        self.recurrent = klass(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class CausalConv1d(nn.Module):
    """Conv1d avec padding uniquement à gauche (jamais le futur)."""

    def __init__(self, channels_in: int, channels_out: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            channels_in, channels_out, kernel_size=3, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(input_channels, output_channels, dilation)
        self.conv2 = CausalConv1d(output_channels, output_channels, dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.activation(self.conv1(x)))
        z = self.dropout(self.activation(self.conv2(z)))
        return self.activation(z + self.residual(x))


class TCNNet(nn.Module):
    """Temporal Convolutional Network résiduel à dilatations causales."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        blocks = []
        in_channels = 1
        for level in range(layers):
            blocks.append(
                TCNBlock(in_channels, hidden, dilation=2 ** level, dropout=dropout)
            )
            in_channels = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input commun (batch, fenêtre, 1) -> (batch, canaux, fenêtre).
        z = self.blocks(x.transpose(1, 2))
        return self.head(z[:, :, -1]).squeeze(-1)


def build_network(
    architecture: str, hidden: int, layers: int, dropout: float
) -> nn.Module:
    """Même entrée et même sortie pour LSTM, GRU et TCN."""
    if hidden < 1 or layers < 1 or not 0 <= dropout < 1:
        raise ValueError("Hyperparamètres du réseau invalides.")
    if architecture in ("LSTM", "GRU"):
        return RecurrentNet(architecture, hidden, layers, dropout)
    if architecture == "TCN":
        return TCNNet(hidden, layers, dropout)
    raise ValueError(f"Architecture inconnue : {architecture}")
