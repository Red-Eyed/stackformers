from __future__ import annotations

import torch

from stackformers.v1.norm.rms import RMSNorm


def test_rms_norm_output_shape() -> None:
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == (2, 16, 64)


def test_rms_norm_unit_vectors() -> None:
    """After RMSNorm the per-token vectors should have L2 norm ≈ sqrt(dim) * g."""
    dim = 32
    norm = RMSNorm(dim=dim)
    x = torch.randn(3, 8, dim)
    out = norm(x)
    norms = out.norm(dim=-1)
    expected = dim**0.5
    assert (norms - expected).abs().max() < 1e-4


def test_rms_norm_learnable_scale() -> None:
    dim = 16
    norm = RMSNorm(dim=dim)
    assert norm.g.shape == (dim,)
    assert norm.g.requires_grad
