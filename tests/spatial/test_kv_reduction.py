from __future__ import annotations

import pytest
import torch

from stackformers.norm.config import LayerNormConfig
from stackformers.spatial.config import ConvKVReductionConfig, NoKVReductionConfig
from stackformers.spatial.factory import build_kv_reduction
from stackformers.spatial.kv_reduction import ConvKVReduction, NoKVReduction
from stackformers.spatial.protocols import KVReduction

B, H, W, D, R = 2, 8, 8, 16, 2
N = H * W


def _conv_reducer(reduction: int = R) -> ConvKVReduction:
    return build_kv_reduction(ConvKVReductionConfig(dim=D, reduction=reduction))  # type: ignore[return-value]


def test_null_and_conv_satisfy_protocol() -> None:
    assert isinstance(NoKVReduction(), KVReduction)
    assert isinstance(_conv_reducer(), KVReduction)


def test_conv_norm_dim_synced_to_dim() -> None:
    config = ConvKVReductionConfig(dim=D, reduction=R, norm=LayerNormConfig(dim=1))
    assert config.norm.dim == D  # validator overrides the placeholder dim


def test_no_kv_reduction_is_identity(device: torch.device) -> None:
    x = torch.randn(B, N, D, device=device)
    ctx, positions = NoKVReduction()(x, grid=(H, W))
    assert torch.equal(ctx, x)
    assert positions.shape == (B, N, 2)


def test_conv_kv_reduction_shapes(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    reducer = _conv_reducer().to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    ctx, positions = reducer(x, grid=(H, W))
    s = (H // R) * (W // R)
    assert ctx.shape == (B, s, D)
    assert positions.shape == (B, s, 2)


def test_conv_kv_reduction_rejects_indivisible_grid(device: torch.device) -> None:
    reducer = _conv_reducer(reduction=3).to(device=device)
    x = torch.randn(B, N, D, device=device)
    with pytest.raises(ValueError, match="not divisible"):
        reducer(x, grid=(H, W))


def test_build_kv_reduction_dispatch() -> None:
    assert isinstance(build_kv_reduction(NoKVReductionConfig()), NoKVReduction)
    assert isinstance(
        build_kv_reduction(ConvKVReductionConfig(dim=D, reduction=R)), ConvKVReduction
    )


def test_conv_kv_reduction_gradients(device: torch.device) -> None:
    reducer = _conv_reducer().to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    ctx, _ = reducer(x, grid=(H, W))
    ctx.sum().backward()
    assert x.grad is not None
