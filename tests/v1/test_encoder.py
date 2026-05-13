from __future__ import annotations

import pytest
import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.config import AttentionConfig
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.encoder import Encoder
from stackformers.v1.feedforward.config import FeedForwardConfig
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.protocols import PosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.sequence import PaddedSequence, make_padded

B, N, D, H, DH = 2, 16, 64, 4, 16
NUM_LAYERS = 3


def _build_encoder(
    device: torch.device,
    dtype: torch.dtype,
    pos_encoding: PosEncoding | None = None,
) -> Encoder:
    attn_cfg = AttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = FeedForwardConfig(dim=D)
    pos: PosEncoding = pos_encoding if pos_encoding is not None else NoPosEncoding()
    layers: list[TransformerLayer] = [
        TransformerLayer(
            self_attn=SelfAttention(attn_cfg, pos, NoBiasBuilder(), SDPAKernel()),
            ff=SwiGLU(ff_cfg),
            norm_attn=RMSNorm(D),
            norm_ff=RMSNorm(D),
        )
        for _ in range(NUM_LAYERS)
    ]
    return Encoder(layers=layers, final_norm=RMSNorm(D)).to(device=device, dtype=dtype)


@pytest.fixture
def encoder(device_dtype: tuple[torch.device, torch.dtype]) -> Encoder:
    device, dtype = device_dtype
    return _build_encoder(device, dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, PaddedSequence]:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    return x, seq


def test_encoder_output_shape(
    encoder: Encoder,
    x_pad: tuple[torch.Tensor, PaddedSequence],
) -> None:
    x, seq = x_pad
    assert encoder(x, seq).shape == (B, N, D)


def test_encoder_with_padding(
    encoder: Encoder,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 12:] = False
    assert encoder(x, make_padded(mask)).shape == (B, N, D)


def test_encoder_with_rope(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    enc = _build_encoder(device, dtype, pos_encoding=RotaryEmbedding1D(dim_head=DH))
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    assert enc(x, seq).shape == (B, N, D)


def test_encoder_gradients(device: torch.device) -> None:
    enc = _build_encoder(device, torch.float32)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    enc(x, seq).sum().backward()
    assert x.grad is not None
