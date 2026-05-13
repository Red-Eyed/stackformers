from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import AttentionConfig
from stackformers.feedforward.config import FeedForwardConfig
from stackformers.norm.config import RMSNormConfig
from stackformers.presets.cross_attender import CrossAttender, CrossAttenderConfig
from stackformers.sequence import make_padded_input

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


@pytest.fixture
def config() -> CrossAttenderConfig:
    return CrossAttenderConfig(
        attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=2,
    )


@pytest.fixture
def cross_attender(
    config: CrossAttenderConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> CrossAttender:
    device, dtype = device_dtype
    return CrossAttender(config).to(device=device, dtype=dtype)


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


def test_cross_attender_output_shape(
    cross_attender: CrossAttender,
    x_context_inp: tuple[object, object],
) -> None:
    x_inp, ctx_inp = x_context_inp
    out = cross_attender(x_inp, ctx_inp)  # type: ignore[arg-type]
    assert out.shape == (B, N, D)


def test_cross_attender_with_ctx_padding(
    cross_attender: CrossAttender,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[1, 8:] = False
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, mask)
    out = cross_attender(x_inp, ctx_inp)
    assert out.shape == (B, N, D)


def test_cross_attender_with_x_padding(
    cross_attender: CrossAttender,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 5:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    out = cross_attender(x_inp, ctx_inp)
    assert out.shape == (B, N, D)


def test_cross_attender_gradients(device: torch.device) -> None:
    cfg = CrossAttenderConfig(
        attn=AttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=2,
    )
    model = CrossAttender(cfg).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    context = torch.randn(B, S, D, device=device, requires_grad=True)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    model(x_inp, ctx_inp).sum().backward()
    assert x.grad is not None
    assert context.grad is not None


def test_cross_attender_config_accessor(config: CrossAttenderConfig) -> None:
    model = CrossAttender(config)
    assert model.config is config


def test_cross_attender_causal_forced_false() -> None:
    causal_cfg = CrossAttenderConfig(
        attn=AttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        ff=FeedForwardConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=1,
    )
    model = CrossAttender(causal_cfg)
    x = torch.randn(B, N, D)
    context = torch.randn(B, S, D)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool))
    out = model(x_inp, ctx_inp)
    assert out.shape == (B, N, D)
