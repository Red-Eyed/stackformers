from __future__ import annotations

import pytest
import torch
import torch.export
from torch import Tensor

from stackformers.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.attention.kernels import (
    SDPAKernel,
    VarlenSDPAKernel,
    VarlenWindowedSDPAKernel,
    WindowedSDPAKernel,
)
from stackformers.sequence import PackedSequence, PaddedSequence, make_packed, make_padded
from tests.conftest import atol

B, H, N, DH = 2, 4, 16, 32


@pytest.fixture
def padded_seq(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedSequence:
    device, _ = device_dtype
    return make_padded(torch.ones(B, N, dtype=torch.bool, device=device))


@pytest.fixture
def packed_seq(device_dtype: tuple[torch.device, torch.dtype]) -> PackedSequence:
    device, _ = device_dtype
    return make_packed(torch.tensor([0, 6, 10], dtype=torch.int32, device=device), max_seqlen=6)


@pytest.fixture
def qkv_padded(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[Tensor, Tensor, Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
    )


@pytest.fixture
def qkv_packed(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[Tensor, Tensor, Tensor]:
    device, dtype = device_dtype
    total = 10
    return (
        torch.randn(total, H, DH, device=device, dtype=dtype),
        torch.randn(total, H, DH, device=device, dtype=dtype),
        torch.randn(total, H, DH, device=device, dtype=dtype),
    )


# --- bias builders ---


def test_no_bias_builder_returns_none(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    assert NoBiasBuilder().forward(N, N, device) is None


def test_alibi_output_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=H).forward(N, N, device)
    assert bias is not None
    assert bias.shape == (H, N, N)


def test_alibi_is_symmetric_non_causal(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=H).forward(N, N, device)
    assert bias is not None
    assert torch.allclose(bias, bias.transpose(-1, -2))


def test_alibi_slopes_differ_across_heads(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=4).forward(N, N, device)
    assert bias is not None
    assert not torch.allclose(bias[0], bias[1])


def test_alibi_non_power_of_two_heads(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=6).forward(N, N, device)
    assert bias is not None
    assert bias.shape == (6, N, N)


# --- SDPA kernel ---


def test_sdpa_kernel_output_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = SDPAKernel()(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_with_bias(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    bias = torch.zeros(H, N, N, device=q.device, dtype=q.dtype)
    out = SDPAKernel()(q, k, v, padded_seq, padded_seq, bias)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_causal(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = SDPAKernel(causal=True)(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_no_k_seq_info(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = SDPAKernel()(q, k, v, padded_seq, None, None)
    assert out.shape == (B, H, N, DH)


# --- windowed kernel ---


def test_windowed_kernel_large_window_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=64)(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_active_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=4)(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_causal_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=4, causal=True)(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_causal_masks_future(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    kernel = WindowedSDPAKernel(window_size=4, causal=True)
    q = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    k = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    v = (
        torch.eye(8, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, 8, 8)[..., :4]
    )
    q[0, 0, 0] = 1.0
    seq = make_padded(torch.ones(1, 8, dtype=torch.bool, device=device))
    out = kernel(q, k, v, seq, seq, None)
    assert out.shape == (1, 1, 8, 4)
    # Token 0 is causal: window covers only itself. With k=zeros, softmax=1.0 on key 0.
    # If future tokens leaked, out[0,0,0] would be a mixture of multiple v rows.
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=atol(dtype))


# --- windowed kernel (unfold mode) ---


def test_windowed_unfold_kernel_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=4, mode="unfold")(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_windowed_unfold_kernel_large_window_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=64, mode="unfold")(q, k, v, padded_seq, padded_seq, None)
    assert out.shape == (B, H, N, DH)


def test_windowed_unfold_kernel_causal_shape(
    qkv_padded: tuple[Tensor, Tensor, Tensor], padded_seq: PaddedSequence
) -> None:
    q, k, v = qkv_padded
    out = WindowedSDPAKernel(window_size=4, causal=True, mode="unfold")(
        q, k, v, padded_seq, padded_seq, None
    )
    assert out.shape == (B, H, N, DH)


def test_windowed_unfold_kernel_causal_masks_future(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    kernel = WindowedSDPAKernel(window_size=4, causal=True, mode="unfold")
    q = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    k = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    v = (
        torch.eye(8, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, 8, 8)[..., :4]
    )
    q[0, 0, 0] = 1.0
    seq = make_padded(torch.ones(1, 8, dtype=torch.bool, device=device))
    out = kernel(q, k, v, seq, seq, None)
    assert out.shape == (1, 1, 8, 4)
    # Same invariant as mask mode: token 0's causal window contains only itself.
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=atol(dtype))


def test_windowed_unfold_matches_mask(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    v = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    out_mask = WindowedSDPAKernel(window_size=4)(q, k, v, seq, seq, None)
    out_unfold = WindowedSDPAKernel(window_size=4, mode="unfold")(q, k, v, seq, seq, None)
    assert torch.allclose(out_mask, out_unfold, atol=atol(dtype))


def test_windowed_unfold_causal_matches_mask(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    v = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    out_mask = WindowedSDPAKernel(window_size=4, causal=True)(q, k, v, seq, seq, None)
    out_unfold = WindowedSDPAKernel(window_size=4, causal=True, mode="unfold")(
        q, k, v, seq, seq, None
    )
    assert torch.allclose(out_mask, out_unfold, atol=atol(dtype))


def test_windowed_unfold_padding_blocks_invalid_keys(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    # Padding keys must not contaminate output: replacing their v with large values must be a no-op.
    device, dtype = device_dtype
    q = torch.randn(1, 1, 8, 4, device=device, dtype=dtype)
    k = torch.randn(1, 1, 8, 4, device=device, dtype=dtype)
    v = torch.randn(1, 1, 8, 4, device=device, dtype=dtype)
    # First 4 keys are valid; keys 4-7 are padding.
    key_mask = torch.tensor([[True, True, True, True, False, False, False, False]], device=device)
    seq = make_padded(key_mask)
    out = WindowedSDPAKernel(window_size=4, mode="unfold")(q, k, v, seq, seq, None)
    v_poisoned = v.clone()
    v_poisoned[0, 0, 4:] = 1e4
    out_poisoned = WindowedSDPAKernel(window_size=4, mode="unfold")(
        q, k, v_poisoned, seq, seq, None
    )
    # Query positions 0-3 only attend to valid keys — output must be identical.
    assert torch.allclose(out[..., :4, :], out_poisoned[..., :4, :], atol=atol(dtype))


def test_windowed_unfold_with_bias_matches_mask(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    v = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    seq = make_padded(torch.ones(B, N, dtype=torch.bool, device=device))
    bias = torch.randn(H, N, N, device=device, dtype=dtype)
    out_mask = WindowedSDPAKernel(window_size=4)(q, k, v, seq, seq, bias)
    out_unfold = WindowedSDPAKernel(window_size=4, mode="unfold")(q, k, v, seq, seq, bias)
    assert torch.allclose(out_mask, out_unfold, atol=atol(dtype))


# --- varlen SDPA kernel ---


def test_varlen_kernel_shape(
    qkv_packed: tuple[Tensor, Tensor, Tensor], packed_seq: PackedSequence
) -> None:
    q, k, v = qkv_packed
    out = VarlenSDPAKernel()(q, k, v, packed_seq, packed_seq, None)
    assert out.shape == q.shape


def test_varlen_kernel_causal_shape(
    qkv_packed: tuple[Tensor, Tensor, Tensor], packed_seq: PackedSequence
) -> None:
    q, k, v = qkv_packed
    out = VarlenSDPAKernel(causal=True)(q, k, v, packed_seq, packed_seq, None)
    assert out.shape == q.shape


# --- varlen windowed kernel ---


def test_varlen_windowed_kernel_shape(
    qkv_packed: tuple[Tensor, Tensor, Tensor], packed_seq: PackedSequence
) -> None:
    q, k, v = qkv_packed
    out = VarlenWindowedSDPAKernel(window_size=4)(q, k, v, packed_seq, packed_seq, None)
    assert out.shape == q.shape


def test_varlen_windowed_kernel_causal_shape(
    qkv_packed: tuple[Tensor, Tensor, Tensor], packed_seq: PackedSequence
) -> None:
    q, k, v = qkv_packed
    out = VarlenWindowedSDPAKernel(window_size=3, causal=True)(
        q, k, v, packed_seq, packed_seq, None
    )
    assert out.shape == q.shape


# --- torch.export ---


def _export_windowed(kernel: WindowedSDPAKernel, n: int) -> torch.export.ExportedProgram:
    kernel.eval()
    q = torch.randn(1, H, n, DH)
    k = torch.randn(1, H, n, DH)
    v = torch.randn(1, H, n, DH)
    seq = make_padded(torch.ones(1, n, dtype=torch.bool))
    n_dim = torch.export.Dim("n", min=1, max=512)
    seq_shapes = {2: n_dim}
    mask_shapes = {1: n_dim}
    return torch.export.export(
        kernel,
        (q, k, v, seq, seq, None),
        dynamic_shapes=(
            seq_shapes,
            seq_shapes,
            seq_shapes,
            PaddedSequence(mask_shapes),  # type: ignore[arg-type]
            PaddedSequence(mask_shapes),  # type: ignore[arg-type]
            None,
        ),
    )


def test_windowed_kernel_export_succeeds() -> None:
    ep = _export_windowed(WindowedSDPAKernel(window_size=4), n=8)
    assert ep is not None


def test_windowed_kernel_export_shorter_than_window() -> None:
    ep = _export_windowed(WindowedSDPAKernel(window_size=4), n=8)
    q = torch.randn(1, H, 3, DH)
    seq = make_padded(torch.ones(1, 3, dtype=torch.bool))
    out = ep.module()(q, q, q, seq, seq, None)
    assert out.shape == (1, H, 3, DH)


def test_windowed_kernel_export_longer_than_traced() -> None:
    ep = _export_windowed(WindowedSDPAKernel(window_size=4), n=8)
    q = torch.randn(1, H, 32, DH)
    seq = make_padded(torch.ones(1, 32, dtype=torch.bool))
    out = ep.module()(q, q, q, seq, seq, None)
    assert out.shape == (1, H, 32, DH)


def test_windowed_causal_kernel_export_succeeds() -> None:
    ep = _export_windowed(WindowedSDPAKernel(window_size=4, causal=True), n=8)
    assert ep is not None
