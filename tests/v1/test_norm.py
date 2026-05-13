from __future__ import annotations

import pytest
import torch

from stackformers.v1.norm.rms import RMSNorm

DIM = 32


@pytest.fixture
def rms_norm(device_dtype: tuple[torch.device, torch.dtype]) -> RMSNorm:
    device, dtype = device_dtype
    return RMSNorm(dim=DIM).to(device=device, dtype=dtype)


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
