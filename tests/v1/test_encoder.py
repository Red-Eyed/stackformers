from __future__ import annotations

import torch

from stackformers.v1.attention.bias import NoBiasBuilder
from stackformers.v1.attention.kernels import SDPAKernel
from stackformers.v1.attention.self_attn import SelfAttention
from stackformers.v1.configs import AttentionConfig, FeedForwardConfig
from stackformers.v1.encoder import Encoder
from stackformers.v1.feedforward.swiglu import SwiGLU
from stackformers.v1.layers import TransformerLayer
from stackformers.v1.norm.rms import RMSNorm
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.protocols import PosEncoding
from stackformers.v1.sequence import make_padded

B, N, D, H, DH = 2, 16, 64, 4, 16
NUM_LAYERS = 3


def _make_encoder(pos_encoding: PosEncoding | None = None) -> Encoder:
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
    return Encoder(layers=layers, final_norm=RMSNorm(D))


def test_encoder_output_shape() -> None:
    enc = _make_encoder()
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = enc(x, seq)
    assert out.shape == (B, N, D)


def test_encoder_with_padding() -> None:
    enc = _make_encoder()
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 12:] = False
    out = enc(x, make_padded(mask))
    assert out.shape == (B, N, D)


def test_encoder_with_rope() -> None:
    enc = _make_encoder(pos_encoding=RotaryEmbedding1D(dim_head=DH))
    x = torch.randn(B, N, D)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    out = enc(x, seq)
    assert out.shape == (B, N, D)


def test_encoder_gradients() -> None:
    enc = _make_encoder()
    x = torch.randn(B, N, D, requires_grad=True)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool))
    enc(x, seq).sum().backward()
    assert x.grad is not None
