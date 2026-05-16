from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.rms import RMSNorm
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.protocols import PosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import PaddedInput, make_padded_input

B, N, D, H, DH = 2, 16, 64, 4, 16
NUM_LAYERS = 3


def _build_encoder(
    device: torch.device,
    dtype: torch.dtype,
    pos_encoding: PosEncoding | None = None,
) -> Encoder:
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    pos: PosEncoding = pos_encoding if pos_encoding is not None else NoPosEncoding()
    norm_cfg = RMSNormConfig(dim=D)
    layers: list[TransformerLayer] = [
        TransformerLayer(
            self_attn=SelfAttention(attn_cfg, pos),
            ff=SwiGLU(ff_cfg),
            norm_attn=RMSNorm(norm_cfg),
            norm_ff=RMSNorm(norm_cfg),
        )
        for _ in range(NUM_LAYERS)
    ]
    return Encoder(layers=layers, final_norm=RMSNorm(norm_cfg)).to(device=device, dtype=dtype)


@pytest.fixture
def encoder(device_dtype: tuple[torch.device, torch.dtype]) -> Encoder:
    device, dtype = device_dtype
    return _build_encoder(device, dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_encoder_output_shape(
    encoder: Encoder,
    x_pad: PaddedInput,
) -> None:
    assert encoder(x_pad).shape == (B, N, D)


def test_encoder_with_padding(
    encoder: Encoder,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 12:] = False
    assert encoder(make_padded_input(x, mask)).shape == (B, N, D)


def test_encoder_with_rope(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    enc = _build_encoder(device, dtype, pos_encoding=RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)))
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    assert enc(make_padded_input(x, mask)).shape == (B, N, D)


def test_encoder_gradients(device: torch.device) -> None:
    enc = _build_encoder(device, torch.float32)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    enc(make_padded_input(x, mask)).sum().backward()
    assert x.grad is not None
