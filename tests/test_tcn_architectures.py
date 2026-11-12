"""Tests locaux : TCN causal, entraînement et compatibilité des checkpoints."""
from io import BytesIO

import pytest
import torch

from src.dl_architectures import CausalConv1d, build_network


@pytest.mark.parametrize("architecture", ["LSTM", "GRU", "TCN"])
def test_architectures_train_and_reload(architecture):
    torch.manual_seed(42)
    network = build_network(architecture, hidden=8, layers=2, dropout=0.1)
    data = torch.randn(5, 30, 1)
    forecasts = network(data)
    assert forecasts.shape == (5,)

    loss = (forecasts - torch.randn(5)).square().mean()
    loss.backward()
    assert any(p.grad is not None for p in network.parameters())

    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)
    buffer.seek(0)
    weights = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = build_network(architecture, hidden=8, layers=2, dropout=0.1)
    restored.load_state_dict(weights)

    network.eval()
    restored.eval()
    torch.testing.assert_close(network(data), restored(data))


def test_causal_conv_never_reads_later_time_steps():
    torch.manual_seed(13)
    conv = CausalConv1d(1, 8, dilation=4)
    before = torch.randn(2, 1, 20)
    modified = before.clone()
    modified[:, :, 10:] = torch.randn_like(modified[:, :, 10:])
    torch.testing.assert_close(
        conv(before)[:, :, :10], conv(modified)[:, :, :10]
    )


def test_tcn_features_do_not_read_later_time_steps():
    torch.manual_seed(24)
    network = build_network("TCN", hidden=8, layers=3, dropout=0.0)
    before = torch.randn(2, 30, 1)
    modified = before.clone()
    modified[:, 10:] = torch.randn_like(modified[:, 10:])

    network.eval()
    torch.testing.assert_close(
        network.blocks(before.transpose(1, 2))[:, :, :10],
        network.blocks(modified.transpose(1, 2))[:, :, :10],
    )
