from __future__ import annotations

import pytest
import torch

from stackformers.presets.encoder import (
    TransformerEncoder,
    packed_encoder_config,
    plain_encoder_config,
    windowed_encoder_config,
)
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input

B, N, D, H = 2, 16, 64, 4
NT = 10  # total tokens for packed (two seqs: 6 + 4)


@pytest.fixture
def padded_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def packed_input(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


def test_plain_encoder_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_windowed_encoder_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = windowed_encoder_config(D, H, num_layers=2, window_size=4)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_packed_encoder_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    cfg = packed_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(packed_input)
    assert out.shape == (NT, D)


def test_plain_encoder_causal(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2, causal=True)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)
