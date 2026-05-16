from __future__ import annotations

import pytest
import torch

from stackformers.norm.config import LayerNormConfig, RMSNormConfig
from stackformers.norm.layer_norm import LayerNorm
from stackformers.norm.rms import RMSNorm

DIM = 32


@pytest.fixture
def rms_norm(device_dtype: tuple[torch.device, torch.dtype]) -> RMSNorm:
    device, dtype = device_dtype
    return RMSNorm(RMSNormConfig(dim=DIM)).to(device=device, dtype=dtype)


@pytest.fixture
def x(device_dtype: tuple[torch.device, torch.dtype]) -> torch.Tensor:
    device, dtype = device_dtype
    return torch.randn(3, 8, DIM, device=device, dtype=dtype)


def test_rms_norm_output_shape(rms_norm: RMSNorm, x: torch.Tensor) -> None:
    assert rms_norm(x).shape == x.shape


def test_rms_norm_unit_vectors(
    rms_norm: RMSNorm,
    x: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    out = rms_norm(x)
    norms = out.norm(dim=-1).float()  # upcast for allclose stability
    expected = DIM**0.5
    assert (norms - expected).abs().max() < (0.5 if dtype == torch.float16 else 1e-4)


def test_rms_norm_learnable_scale(rms_norm: RMSNorm) -> None:
    assert rms_norm.g.shape == (DIM,)
    assert rms_norm.g.requires_grad


# --- LayerNorm ---


@pytest.fixture
def layer_norm(device_dtype: tuple[torch.device, torch.dtype]) -> LayerNorm:
    device, dtype = device_dtype
    return LayerNorm(LayerNormConfig(dim=DIM)).to(device=device, dtype=dtype)


def test_layer_norm_output_shape(layer_norm: LayerNorm, x: torch.Tensor) -> None:
    assert layer_norm(x).shape == x.shape


def test_layer_norm_zero_mean(
    layer_norm: LayerNorm,
    x: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    out = layer_norm(x).float()
    means = out.mean(dim=-1).abs()
    tol = 5e-2 if dtype == torch.float16 else 1e-5
    assert means.max() < tol


def test_layer_norm_unit_variance(
    layer_norm: LayerNorm,
    x: torch.Tensor,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    out = layer_norm(x).float()
    var = out.var(dim=-1, unbiased=False)
    tol = 5e-2 if dtype == torch.float16 else 1e-4
    assert (var - 1.0).abs().max() < tol


def test_layer_norm_learnable_params(layer_norm: LayerNorm) -> None:
    params = dict(layer_norm.named_parameters())
    assert "norm.weight" in params
    assert "norm.bias" in params


def test_layer_norm_gradients_flow(device: torch.device) -> None:
    norm = LayerNorm(LayerNormConfig(dim=DIM)).to(device=device)
    x = torch.randn(3, 8, DIM, device=device, requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None
