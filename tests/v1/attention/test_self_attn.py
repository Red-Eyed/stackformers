from __future__ import annotations

import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.configs import AttentionConfig
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.sequence import PaddedSequence, make_padded

B, N, D, H, DH = 2, 16, 64, 4, 16


def _make_self_attn(causal: bool = False) -> SelfAttention:
    config = AttentionConfig(dim=D, heads=H, dim_head=DH, causal=causal)
    return SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    )


def test_self_attn_output_shape() -> None:
    attn = _make_self_attn()
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = attn(x, seq)
    assert out.shape == (B, N, D)


def test_self_attn_causal_shape() -> None:
    attn = _make_self_attn(causal=True)
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = attn(x, seq)
    assert out.shape == (B, N, D)


def test_self_attn_with_rope() -> None:
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = SelfAttention(
        config=config,
        pos_encoding=RotaryEmbedding1D(dim_head=DH),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    )
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = attn(x, seq)
    assert out.shape == (B, N, D)


def test_self_attn_with_padding_mask() -> None:
    attn = _make_self_attn()
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 12:] = False
    seq = PaddedSequence(mask=mask)
    out = attn(x, seq)
    assert out.shape == (B, N, D)


def test_self_attn_gqa() -> None:
    """Grouped query attention: kv_heads < heads."""
    config = AttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2)
    attn = SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        bias_builder=NoBiasBuilder(),
        kernel=SDPAKernel(),
    )
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = attn(x, seq)
    assert out.shape == (B, N, D)
