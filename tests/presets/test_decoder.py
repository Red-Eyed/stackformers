from __future__ import annotations

import pytest
import torch

from stackformers.presets.decoder import TransformerDecoder, plain_decoder_config
from stackformers.sequence import PaddedInput, make_padded_input

B, N, S, D, H = 2, 8, 12, 64, 4  # N=target len, S=context len


@pytest.fixture
def target_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def context_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_plain_decoder_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    target_input: PaddedInput,
    context_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_decoder_config(D, H, num_layers=2)
    out = TransformerDecoder(cfg).to(device=device, dtype=dtype)(target_input, context_input)
    assert out.shape == (B, N, D)
