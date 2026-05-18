"""Windowed and global self-attention behavioral tests.

Previously this file tested kernel classes (SDPAKernel, WindowedSDPAKernel, etc.).
Those classes were removed — math is now inline in SelfAttention.
Tests here verify the behavioral contracts: correct shapes for all combinations
of global/windowed × causal/non-causal × padded/packed, and causal mask correctness.
"""

from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PackedInput, PaddedInput, make_packed_input, make_padded_input
from tests.conftest import atol

B, N, D, H, DH = 2, 16, 64, 4, 16
NT = 10  # two packed seqs: 6 + 4


def _attn(
    device: torch.device,
    dtype: torch.dtype,
    *,
    window_size: int | None = None,
    causal: bool = False,
) -> SelfAttention:
    cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, window_size=window_size, causal=causal)
    return SelfAttention(cfg, NoPosEncoding()).to(device=device, dtype=dtype).eval()


@pytest.fixture
def padded_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    return make_padded_input(
        torch.randn(B, N, D, device=device, dtype=dtype),
        torch.ones(B, N, dtype=torch.bool, device=device),
    )


@pytest.fixture
def packed_input(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    if not device.type == "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


# --- shape contracts ---


def test_global_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype)(padded_input).shape == (B, N, D)


def test_global_causal_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, causal=True)(padded_input).shape == (B, N, D)


def test_global_packed_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype)(packed_input).shape == (NT, D)


def test_global_causal_packed_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, causal=True)(packed_input).shape == (NT, D)


def test_windowed_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, window_size=4)(padded_input).shape == (B, N, D)


def test_windowed_causal_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, window_size=4, causal=True)(padded_input).shape == (B, N, D)


def test_windowed_packed_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, window_size=4)(packed_input).shape == (NT, D)


def test_windowed_causal_packed_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    packed_input: PackedInput,
) -> None:
    device, dtype = device_dtype
    assert _attn(device, dtype, window_size=4, causal=True)(packed_input).shape == (NT, D)


def test_windowed_large_window_padded_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    padded_input: PaddedInput,
) -> None:
    """Window larger than sequence length — degenerates to global attention."""
    device, dtype = device_dtype
    assert _attn(device, dtype, window_size=64)(padded_input).shape == (B, N, D)


# --- causal masking correctness ---


def test_windowed_causal_masks_future(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Token 0 with a causal window can only attend to itself; future tokens must be invisible."""
    device, dtype = device_dtype
    N8, D8, H1, DH8 = 8, 64, 1, 64
    # Wire q/k projections to zero so attention weights are uniform within the window.
    # Wire v/out projections to identity so output = weighted sum of v = input rows.
    cfg = SelfAttentionConfig(dim=D8, heads=H1, dim_head=DH8, window_size=4, causal=True)
    model = SelfAttention(cfg, NoPosEncoding()).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        model.to_q.weight.zero_()
        model.to_k.weight.zero_()
        model.to_v.weight.copy_(torch.eye(D8, device=device, dtype=dtype))
        model.to_out.weight.copy_(torch.eye(D8, device=device, dtype=dtype))

    x = torch.eye(N8, D8, device=device, dtype=dtype).unsqueeze(0)  # 1 N8 D8
    inp = make_padded_input(x, torch.ones(1, N8, dtype=torch.bool, device=device))
    out = model(inp)
    # Token 0: causal window = [0 - 4, 0] ∩ [0, N8-1] = {0}. Uniform softmax over {0} → v[0].
    assert torch.allclose(out[0, 0], x[0, 0], atol=atol(dtype))


def test_padding_mask_respected(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Replacing padded-position values with large poison values must not change valid outputs."""
    device, dtype = device_dtype
    cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH, window_size=4)
    model = SelfAttention(cfg, NoPosEncoding()).to(device=device, dtype=dtype).eval()
    key_mask = torch.ones(1, N, dtype=torch.bool, device=device)
    key_mask[0, N // 2 :] = False  # second half is padding
    x = torch.randn(1, N, D, device=device, dtype=dtype)
    x_poisoned = x.clone()
    x_poisoned[0, N // 2 :] = 1e4
    with torch.no_grad():
        out = model(make_padded_input(x, key_mask))
        out_poisoned = model(make_padded_input(x_poisoned, key_mask))
    # Valid positions must be unaffected by the poison values in padding positions.
    assert torch.allclose(out[0, : N // 2], out_poisoned[0, : N // 2], atol=atol(dtype))
