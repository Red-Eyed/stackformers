from __future__ import annotations

import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.configs import AttentionConfig, FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.sequence import make_padded

B, N, D, H, DH = 2, 16, 64, 4, 16


def _make_layer() -> TransformerLayer:
    attn_cfg = AttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = FeedForwardConfig(dim=D)
    self_attn = SelfAttention(attn_cfg, NoPosEncoding(), NoBiasBuilder(), SDPAKernel())
    ff = SwiGLU(ff_cfg)
    return TransformerLayer(
        self_attn=self_attn,
        ff=ff,
        norm_attn=RMSNorm(D),
        norm_ff=RMSNorm(D),
    )


def test_transformer_layer_output_shape() -> None:
    layer = _make_layer()
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = layer(x, seq)
    assert out.shape == (B, N, D)


def test_transformer_layer_residual_connection() -> None:
    """Output should differ from input (residual adds to it)."""
    layer = _make_layer()
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = layer(x, seq)
    assert not torch.allclose(out, x)


def test_transformer_layer_gradients() -> None:
    layer = _make_layer()
    x = torch.randn(B, N, D, requires_grad=True)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = layer(x, seq)
    out.sum().backward()
    assert x.grad is not None
