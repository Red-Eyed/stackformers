from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import AttentionConfig
from stackformers.attention.kernels import SDPAKernel
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import PaddedInput, make_padded_input

B, N, D, H, DH = 2, 16, 64, 4, 16


@pytest.fixture
def self_attn(device_dtype: tuple[torch.device, torch.dtype]) -> SelfAttention:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    return SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        kernel=SDPAKernel(),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def self_attn_rope(device_dtype: tuple[torch.device, torch.dtype]) -> SelfAttention:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    return SelfAttention(
        config=config,
        pos_encoding=RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)),
        kernel=SDPAKernel(),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_self_attn_output_shape(
    self_attn: SelfAttention,
    x_pad: PaddedInput,
) -> None:
    out = self_attn(x_pad)
    assert out.shape == (B, N, D)


def test_self_attn_causal_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    x_pad: PaddedInput,
) -> None:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH, causal=True)
    attn = SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        kernel=SDPAKernel(causal=True),
    ).to(device=device, dtype=dtype)
    out = attn(x_pad)
    assert out.shape == (B, N, D)


def test_self_attn_with_rope(
    self_attn_rope: SelfAttention,
    x_pad: PaddedInput,
) -> None:
    out = self_attn_rope(x_pad)
    assert out.shape == (B, N, D)


def test_self_attn_with_padding_mask(
    self_attn: SelfAttention,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[1, 12:] = False
    out = self_attn(make_padded_input(x, mask))
    assert out.shape == (B, N, D)


def test_self_attn_gqa(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    config = AttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2)
    attn = SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        kernel=SDPAKernel(),
    ).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    out = attn(make_padded_input(x, mask))
    assert out.shape == (B, N, D)


def test_self_attn_gradients(device: torch.device) -> None:
    config = AttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = SelfAttention(
        config=config,
        pos_encoding=NoPosEncoding(),
        kernel=SDPAKernel(),
    ).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    attn(make_padded_input(x, mask)).sum().backward()
    assert x.grad is not None
