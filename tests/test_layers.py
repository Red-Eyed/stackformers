from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PaddedInput, make_padded_input

B, N, D, H, DH = 2, 16, 64, 4, 16


@pytest.fixture
def layer(device_dtype: tuple[torch.device, torch.dtype]) -> TransformerLayer:
    device, dtype = device_dtype
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    norm_cfg = RMSNormConfig(dim=D)
    return TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding()),
        ff=SwiGLU(ff_cfg),
        norm_attn=build_norm(norm_cfg),
        norm_ff=build_norm(norm_cfg),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_transformer_layer_output_shape(
    layer: TransformerLayer,
    x_pad: PaddedInput,
) -> None:
    out = layer(x_pad)
    assert out.x.shape == (B, N, D)


def test_transformer_layer_residual_connection(
    layer: TransformerLayer,
    x_pad: PaddedInput,
) -> None:
    out = layer(x_pad)
    assert not torch.allclose(out.x, x_pad.x)


def test_transformer_layer_gradients(device: torch.device) -> None:
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    norm_cfg = RMSNormConfig(dim=D)
    layer = TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding()),
        ff=SwiGLU(ff_cfg),
        norm_attn=build_norm(norm_cfg),
        norm_ff=build_norm(norm_cfg),
    ).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    inp = make_padded_input(x, mask)
    layer(inp).x.sum().backward()
    assert x.grad is not None
