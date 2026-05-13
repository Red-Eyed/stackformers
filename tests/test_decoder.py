from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import AttentionConfig
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.norm.config import RMSNormConfig
from stackformers.positional.config import NoPosEncodingConfig, RoPE1DConfig
from stackformers.presets.decoder import TransformerDecoder, TransformerDecoderConfig
from stackformers.sequence import make_padded_input

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


@pytest.fixture
def config() -> TransformerDecoderConfig:
    return TransformerDecoderConfig(
        self_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        cross_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )


@pytest.fixture
def decoder(
    config: TransformerDecoderConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> TransformerDecoder:
    device, dtype = device_dtype
    return TransformerDecoder(config).to(device=device, dtype=dtype)


@pytest.fixture
def x_context_inp(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[object, object]:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    return x_inp, ctx_inp


def test_decoder_output_shape(
    decoder: TransformerDecoder,
    x_context_inp: tuple[object, object],
) -> None:
    x_inp, ctx_inp = x_context_inp
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)  # type: ignore[arg-type]


def test_decoder_with_ctx_padding(
    decoder: TransformerDecoder,
    x_context_inp: tuple[object, object],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x_inp, _ = x_context_inp
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[1, 8:] = False
    ctx_inp = make_padded_input(context, mask)
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)  # type: ignore[arg-type]


def test_decoder_with_tgt_padding(
    decoder: TransformerDecoder,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 6:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_self_attn_always_causal() -> None:
    cfg = TransformerDecoderConfig(
        self_attn=AttentionConfig(dim=D, heads=H, dim_head=DH, causal=False),
        cross_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=1,
    )
    model = TransformerDecoder(cfg)
    x = torch.randn(B, N, D)
    context = torch.randn(B, S, D)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_rope(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    cfg = TransformerDecoderConfig(
        self_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        cross_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=RoPE1DConfig(dim_head=DH),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_gradients(device: torch.device) -> None:
    cfg = TransformerDecoderConfig(
        self_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        cross_attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    context = torch.randn(B, S, D, device=device, requires_grad=True)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    model(x_inp, ctx_inp).sum().backward()
    assert x.grad is not None
    assert context.grad is not None


def test_decoder_config_accessor(config: TransformerDecoderConfig) -> None:
    assert TransformerDecoder(config).config is config
