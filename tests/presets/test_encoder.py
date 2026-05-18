from __future__ import annotations

import pytest
import torch

from stackformers.presets.encoder import (
    TransformerEncoder,
    plain_encoder_config,
    windowed_encoder_config,
)
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input

B, N, D, H = 2, 16, 64, 4
NT = 10  # two packed seqs: 6 + 4


@pytest.fixture
def padded_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def packed_input(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    if not device.type == "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


def test_plain_encoder_padded_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_plain_encoder_packed_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_encoder_config(D, H, num_layers=2)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(packed_input)
    assert out.shape == (NT, D)


def test_windowed_encoder_padded_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = windowed_encoder_config(D, H, num_layers=2, window_size=4)
    out = TransformerEncoder(cfg).to(device=device, dtype=dtype)(padded_input)
    assert out.shape == (B, N, D)


def test_windowed_encoder_packed_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    cfg = windowed_encoder_config(D, H, num_layers=2, window_size=4)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed attention requires CUDA")
def test_padded_and_packed_share_weights() -> None:
    """Same model weights handle both padded (inference) and packed (training) inputs."""
    cfg = plain_encoder_config(D, H, num_layers=2)
    enc = TransformerEncoder(cfg).to(device="cuda", dtype=torch.float16)
    padded = make_padded_input(
        torch.randn(B, N, D, device="cuda", dtype=torch.float16),
        torch.ones(B, N, dtype=torch.bool, device="cuda"),
    )
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device="cuda")
    packed = make_packed_input(
        torch.randn(NT, D, device="cuda", dtype=torch.float16), cu, max_seqlen=6
    )
    with torch.no_grad():
        padded_out = enc(padded)
        packed_out = enc(packed)
    assert padded_out.shape == (B, N, D)
    assert packed_out.shape == (NT, D)
