from __future__ import annotations

import pytest
import torch

from stackformers.attention.bias import NoBiasBuilder
from stackformers.attention.config import AttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.kernels import SDPAKernel
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import make_padded_input

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
def x_ctx_input(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, N, D, device=device, dtype=dtype),
        torch.randn(B, S, D, device=device, dtype=dtype),
    )


def test_cross_attn_output_shape(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Output keeps x's sequence length and model dim regardless of context length."""
    x, ctx = x_ctx_input
    device = x.device
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(ctx, torch.ones(B, S, dtype=torch.bool, device=device))
    out = cross_attn(x_inp, ctx_inp)
    assert out.shape == (B, N, D)


def test_cross_attn_with_ctx_mask(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Padding on the context (key) side does not crash and preserves output shape."""
    x, ctx = x_ctx_input
    device = x.device
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[0, 10:] = False
    ctx_inp = make_padded_input(ctx, mask)
    out = cross_attn(x_inp, ctx_inp)
    assert out.shape == (B, N, D)


def test_cross_attn_with_x_mask(
    cross_attn: CrossAttention,
    x_ctx_input: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Padding on x zeros out output positions beyond the query mask boundary."""
    x, ctx = x_ctx_input
    device = x.device
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 5:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(ctx, torch.ones(B, S, dtype=torch.bool, device=device))
    out = cross_attn(x_inp, ctx_inp)
    assert out.shape == (B, N, D)
    assert out[0, 5:].eq(0).all()


def test_cross_attn_output_driven_by_context(
    cross_attn: CrossAttention,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """K/V come from context: with x=0 (uniform attention), changing context changes output.

    If K/V were accidentally sourced from x, both calls would return identical results
    because the attention pattern would be the same uniform distribution over x=0.
    """
    device, dtype = device_dtype
    x = torch.zeros(B, N, D, device=device, dtype=dtype)
    ctx1 = torch.randn(B, S, D, device=device, dtype=dtype)
    ctx2 = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx1_inp = make_padded_input(ctx1, torch.ones(B, S, dtype=torch.bool, device=device))
    ctx2_inp = make_padded_input(ctx2, torch.ones(B, S, dtype=torch.bool, device=device))
    with torch.no_grad():
        out1 = cross_attn(x_inp, ctx1_inp)
        out2 = cross_attn(x_inp, ctx2_inp)
    assert not torch.allclose(out1, out2)


def test_cross_attn_different_seq_lengths(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """x and context may have different sequence lengths; output shape follows x."""
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
    x_inp = make_padded_input(x, torch.ones(B, 5, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(ctx, torch.ones(B, 20, dtype=torch.bool, device=device))
    out = attn(x_inp, ctx_inp)
    assert out.shape == (B, 5, D)
