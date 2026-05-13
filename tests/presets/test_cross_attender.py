from __future__ import annotations

import pytest
import torch

from stackformers.presets.cross_attender import CrossAttender, plain_cross_attender_config
from stackformers.sequence import PaddedInput, make_padded_input

B, Nq, S, D, H = 2, 8, 12, 64, 4  # Nq=query len, S=context len


@pytest.fixture
def query_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, Nq, D, device=device, dtype=dtype)
    mask = torch.ones(B, Nq, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def context_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_plain_cross_attender_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    query_input: PaddedInput,
    context_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_cross_attender_config(D, H, num_layers=2)
    out = CrossAttender(cfg).to(device=device, dtype=dtype)(query_input, context_input)
    assert out.shape == (B, Nq, D)
