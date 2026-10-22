from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from stackformers.attention.bias import NoAttnBias
from stackformers.attention.cached import CachedSelfAttentionWrapper
from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.packed_self_attn import PackedSelfAttention
from stackformers.attention.protocols import AttnBias
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import (
    PackedInput,
    PaddedInput,
    make_packed_input,
    make_padded_input,
    materialize_padding_mask,
)
from tests.conftest import atol


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


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("kv_heads", [None, 2, 1])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_cached_self_attn_matches_full_causal_prefix(
    device_dtype: tuple[torch.device, torch.dtype],
    x_pad: PaddedInput,
    use_rope: bool,
    kv_heads: int | None,
    packed: bool,
    qk_norm: bool,
) -> None:
    """Required growing caches preserve causal output with RoPE and GQA."""
    device, dtype = device_dtype
    config = SelfAttentionConfig(
        dim=D,
        heads=H,
        dim_head=DH,
        kv_heads=kv_heads,
        causal=True,
        qk_norm=qk_norm,
    )
    position = RotaryEmbedding1D(RoPE1DConfig(dim_head=DH)) if use_rope else NoPosEncoding()
    attention = SelfAttention(config=config, pos_encoding=position).to(
        device=device,
        dtype=dtype,
    )
    wrapped = attention.to_packed() if packed else attention
    cached = CachedSelfAttentionWrapper(wrapped)
    assert cached.attention is wrapped
    assert {id(p) for p in cached.parameters()} == {id(p) for p in wrapped.parameters()}
    projection_calls = Mock(return_value=None)
    match wrapped:
        case PackedSelfAttention():
            wrapped.qkv.register_forward_hook(projection_calls)
    cache = torch.zeros(
        2,
        B,
        config.effective_kv_heads,
        0,
        DH,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        for step in range(N):
            prefix = PaddedInput(
                *(
                    tensor[:, : step + 1]
                    for tensor in (x_pad.x, materialize_padding_mask(x_pad), x_pad.abs_positions)
                )
            )
            token = PaddedInput(
                *(
                    tensor[:, step : step + 1]
                    for tensor in (x_pad.x, materialize_padding_mask(x_pad), x_pad.abs_positions)
                )
            )
            expected = attention(prefix)[:, -1:]
            actual, cache = cached(
                token,
                cache,
                torch.tensor([step], dtype=torch.int64, device=device),
            )
            torch.testing.assert_close(actual, expected, atol=atol(dtype), rtol=atol(dtype))

    assert cache.shape == (2, B, config.effective_kv_heads, N, DH)
    assert projection_calls.call_count == (N if packed else 0)


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


@pytest.fixture
def mean_attention(device_dtype: tuple[torch.device, torch.dtype]) -> SelfAttention:
    """Make output equal the mean of visible values for transparent numerical checks."""
    device, dtype = device_dtype
    attention = SelfAttention(
        SelfAttentionConfig(dim=4, heads=1, dim_head=4, causal=True, window_size=1),
        NoPosEncoding(),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        attention.to_q.weight.zero_()
        attention.to_k.weight.zero_()
        attention.to_v.weight.copy_(torch.eye(4, device=device, dtype=dtype))
        attention.to_out.weight.copy_(torch.eye(4, device=device, dtype=dtype))
    return attention


@pytest.fixture
def mean_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    """Use distinct document values and conspicuous padding to expose visibility leaks."""
    device, dtype = device_dtype
    values = torch.tensor([[1, 3, 9], [10, 20, 999]], device=device, dtype=dtype)
    valid = torch.tensor([[True, True, True], [True, True, False]], device=device)
    return make_padded_input(values[..., None].expand(-1, -1, 4), valid)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("mode", "expected_values"),
    [
        ("default", [[1, 2, 6], [10, 15, 20]]),
        ("causal", [[1, 2, 13 / 3], [10, 15, 15]]),
        ("bidirectional", [[13 / 3] * 3, [15] * 3]),
        ("custom", [[3, 1, 0], [20, 10, 0]]),
    ],
)
def test_runtime_self_visibility(
    mean_attention: SelfAttention,
    mean_input: PaddedInput,
    mode: Literal["default", "causal", "bidirectional", "custom"],
    expected_values: list[list[float]],
    packed: bool,
) -> None:
    """Every mode has known means, excludes padding, and keeps packed documents separate."""
    from stackformers.sequence import padded_to_packed
    from tests.conftest import atol

    mask = (
        torch.tensor(
            [[False, True, False], [True, False, False], [False, False, False]],
            device=mean_input.x.device,
        )
        if mode == "custom"
        else mode
    )
    input = padded_to_packed(mean_input) if packed else mean_input
    expected = mean_input.x.new_tensor(expected_values)[..., None].expand(-1, -1, 4)
    if packed:
        expected = expected[materialize_padding_mask(mean_input)]
    actual = mean_attention(input, self_attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=0)
    assert mean_attention.config.causal
    assert mean_attention.config.window_size == 1
    torch.testing.assert_close(
        mean_attention(input), mean_attention(input, self_attn_mask="default")
    )
