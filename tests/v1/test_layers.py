from __future__ import annotations

import pytest
import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.feedforward.config import FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.sequence import PaddedSequence, make_padded

B, N, D, H, DH = 2, 16, 64, 4, 16


@pytest.fixture
def layer(device_dtype: tuple[torch.device, torch.dtype]) -> TransformerLayer:
    device, dtype = device_dtype
    attn_cfg = AttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = FeedForwardConfig(dim=D)
    return TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding(), NoBiasBuilder(), SDPAKernel()),
        ff=SwiGLU(ff_cfg),
        norm_attn=RMSNorm(D),
        norm_ff=RMSNorm(D),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, PaddedSequence]:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    return x, seq


def test_transformer_layer_output_shape(
    layer: TransformerLayer,
    x_pad: tuple[torch.Tensor, PaddedSequence],
) -> None:
    x, seq = x_pad
    assert layer(x, seq).shape == (B, N, D)


def test_transformer_layer_residual_connection(
    layer: TransformerLayer,
    x_pad: tuple[torch.Tensor, PaddedSequence],
) -> None:
    x, seq = x_pad
    out = layer(x, seq)
    assert not torch.allclose(out, x)


def test_transformer_layer_gradients(device: torch.device) -> None:
    attn_cfg = AttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = FeedForwardConfig(dim=D)
    layer = TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding(), NoBiasBuilder(), SDPAKernel()),
        ff=SwiGLU(ff_cfg),
        norm_attn=RMSNorm(D),
        norm_ff=RMSNorm(D),
    ).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    layer(x, seq).sum().backward()
    assert x.grad is not None
