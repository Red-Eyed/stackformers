from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stackformers.norm.config import LayerNormConfig, RMSNormConfig
from stackformers.norm.factory import build_norm

DIM = 32


@pytest.fixture
def rms_norm(device_dtype: tuple[torch.device, torch.dtype]) -> nn.RMSNorm:
    device, dtype = device_dtype
    return build_norm(RMSNormConfig(dim=DIM)).to(device=device, dtype=dtype)  # type: ignore[return-value]


@pytest.fixture
def layer_norm(device_dtype: tuple[torch.device, torch.dtype]) -> nn.LayerNorm:
    device, dtype = device_dtype
    return build_norm(LayerNormConfig(dim=DIM)).to(device=device, dtype=dtype)  # type: ignore[return-value]


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return torch.randn(3, 8, DIM, device=device, dtype=dtype)


def test_rms_norm_output_shape(rms_norm: nn.RMSNorm, x: torch.Tensor) -> None:
    assert rms_norm(x).shape == x.shape


def test_layer_norm_output_shape(layer_norm: nn.LayerNorm, x: torch.Tensor) -> None:
    assert layer_norm(x).shape == x.shape
