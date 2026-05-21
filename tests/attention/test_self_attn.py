from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.protocols import AttnBias
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input


class ConstantBias(nn.Module):
    """AttnBias that adds a fixed scalar to every logit — useful for testing."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def __call__(self, input: PaddedInput) -> torch.Tensor:
        b, n = input.x.shape[:2]
        return torch.full((b, 1, n, n), self.value, device=input.x.device, dtype=input.x.dtype)


B, N, D, H, DH = 2, 16, 64, 4, 16
NT = 10  # two packed seqs: 6 + 4


@pytest.fixture
def self_attn(device_dtype: tuple[torch.device, torch.dtype]) -> SelfAttention:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    return SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)


@pytest.fixture
def self_attn_rope(device_dtype: tuple[torch.device, torch.dtype]) -> SelfAttention:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    return SelfAttention(
        config=config,
        pos_encoding=RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def x_packed(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    if not device.type == "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


def test_self_attn_padded_output_shape(
    self_attn: SelfAttention,
    x_pad: PaddedInput,
) -> None:
    assert self_attn(x_pad).shape == (B, N, D)


def test_self_attn_packed_output_shape(
    self_attn: SelfAttention,
    x_packed: PackedInput,
) -> None:
    assert self_attn(x_packed).shape == (NT, D)


def test_self_attn_causal_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    x_pad: PaddedInput,
) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True)
    attn = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)
    assert attn(x_pad).shape == (B, N, D)


def test_self_attn_windowed_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    x_pad: PaddedInput,
) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, window_size=4)
    attn = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)
    assert attn(x_pad).shape == (B, N, D)


def test_self_attn_windowed_packed_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    x_packed: PackedInput,
) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, window_size=4)
    attn = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)
    assert attn(x_packed).shape == (NT, D)


def test_self_attn_with_rope(
    self_attn_rope: SelfAttention,
    x_pad: PaddedInput,
) -> None:
    assert self_attn_rope(x_pad).shape == (B, N, D)


def test_self_attn_with_padding_mask(
    self_attn: SelfAttention,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[1, 12:] = False
    assert self_attn(make_padded_input(x, mask)).shape == (B, N, D)


def test_self_attn_gqa(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2)
    attn = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    assert attn(make_padded_input(x, mask)).shape == (B, N, D)


def test_self_attn_gradients(device: torch.device) -> None:
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    attn(make_padded_input(x, mask)).sum().backward()
    assert x.grad is not None


# ── AttnBias tests ────────────────────────────────────────────────────────────


def test_no_attn_bias_satisfies_protocol() -> None:
    assert isinstance(NoAttnBias(), AttnBias)


def test_custom_bias_satisfies_protocol() -> None:
    assert isinstance(ConstantBias(0.0), AttnBias)


def test_attn_bias_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    x_pad: PaddedInput,
) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    attn = SelfAttention(
        config=config, pos_encoding=NoPosEncoding(), attn_bias=ConstantBias(0.0)
    ).to(device=device, dtype=dtype)
    assert attn(x_pad).shape == (B, N, D)


def test_attn_bias_changes_output(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    config = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    x = torch.randn(1, N, D, device=device, dtype=dtype)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    inp = make_padded_input(x, mask)

    base = SelfAttention(config=config, pos_encoding=NoPosEncoding()).to(device=device, dtype=dtype)
    biased = SelfAttention(
        config=config, pos_encoding=NoPosEncoding(), attn_bias=ConstantBias(1e4)
    ).to(device=device, dtype=dtype)
    # copy weights so the only difference is the bias
    biased.load_state_dict(base.state_dict())

    with torch.no_grad():
        out_base = base(inp)
        out_biased = biased(inp)

    assert not torch.allclose(out_base, out_biased)
