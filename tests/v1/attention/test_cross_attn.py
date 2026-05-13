from __future__ import annotations

import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.cross_attn import CrossAttention
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.configs import AttentionConfig
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.sequence import make_padded

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


def _make_cross_attn() -> CrossAttention:
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    return CrossAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    )


def test_cross_attn_output_shape() -> None:
    attn = _make_cross_attn()
    x = torch.randn(B, N, D)
    ctx = torch.randn(B, S, D)
    out = attn(x, ctx)
    assert out.shape == (B, N, D)


def test_cross_attn_with_ctx_mask() -> None:
    attn = _make_cross_attn()
    x = torch.randn(B, N, D)
    ctx = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[0, 10:] = False
    ctx_seq = make_padded(mask)
    out = attn(x, ctx, ctx_seq_info=ctx_seq)
    assert out.shape == (B, N, D)


def test_cross_attn_different_seq_lengths() -> None:
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = CrossAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    )
    x = torch.randn(B, 5, D)
    ctx = torch.randn(B, 20, D)
    out = attn(x, ctx)
    assert out.shape == (B, 5, D)
