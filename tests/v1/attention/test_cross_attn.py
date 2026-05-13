from __future__ import annotations

import pytest
import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.attention.cross_attn import CrossAttention
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.sequence import make_padded

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


@pytest.fixture
def cross_attn(device_dtype: tuple[torch.device, torch.dtype]) -> CrossAttention:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    return CrossAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_ctx(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, N, D, device=device, dtype=dtype),
        torch.randn(B, S, D, device=device, dtype=dtype),
    )


def test_cross_attn_output_shape(
    cross_attn: CrossAttention,
    x_ctx: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, ctx = x_ctx
    out = cross_attn(x, ctx)
    assert out.shape == (B, N, D)


def test_cross_attn_with_ctx_mask(
    cross_attn: CrossAttention,
    x_ctx: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, ctx = x_ctx
    device = x.device
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[0, 10:] = False
    out = cross_attn(x, ctx, ctx_seq_info=make_padded(mask))
    assert out.shape == (B, N, D)


def test_cross_attn_different_seq_lengths(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = CrossAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    ).to(device=device, dtype=dtype)
    x = torch.randn(B, 5, D, device=device, dtype=dtype)
    ctx = torch.randn(B, 20, D, device=device, dtype=dtype)
    out = attn(x, ctx)
    assert out.shape == (B, 5, D)
