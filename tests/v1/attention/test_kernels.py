from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from stackformers.v1.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.v1.attention.kernels import (
    SDPAKernel,
    VarlenSDPAKernel,
    VarlenWindowedSDPAKernel,
    WindowedSDPAKernel,
)

B, H, N, DH = 2, 4, 16, 32


class QKV(NamedTuple):
    q: Tensor
    k: Tensor
    v: Tensor


class VarlenBatch(NamedTuple):
    q: Tensor
    k: Tensor
    v: Tensor
    cu_seqlens: Tensor
    max_seqlen: int


@pytest.fixture
def qkv(device_dtype: tuple[torch.device, torch.dtype]) -> QKV:
    device, dtype = device_dtype
    return QKV(
        q=torch.randn(B, H, N, DH, device=device, dtype=dtype),
        k=torch.randn(B, H, N, DH, device=device, dtype=dtype),
        v=torch.randn(B, H, N, DH, device=device, dtype=dtype),
    )


@pytest.fixture
def varlen_batch(device_dtype: tuple[torch.device, torch.dtype]) -> VarlenBatch:
    device, dtype = device_dtype
    total = 10
    return VarlenBatch(
        q=torch.randn(total, H, DH, device=device, dtype=dtype),
        k=torch.randn(total, H, DH, device=device, dtype=dtype),
        v=torch.randn(total, H, DH, device=device, dtype=dtype),
        cu_seqlens=torch.tensor([0, 6, 10], dtype=torch.int32, device=device),
        max_seqlen=6,
    )


@pytest.fixture
def sdpa_kernel() -> SDPAKernel:
    return SDPAKernel()


@pytest.fixture
def varlen_kernel() -> VarlenSDPAKernel:
    return VarlenSDPAKernel()


@pytest.fixture
def varlen_causal_kernel() -> VarlenSDPAKernel:
    return VarlenSDPAKernel(causal=True)


# --- bias builders ---


def test_no_bias_builder_returns_none(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    result = NoBiasBuilder().forward(N, N, device)
    assert result is None


def test_alibi_output_shape(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=H).forward(N, N, device)
    assert bias is not None
    assert bias.shape == (H, N, N)


def test_alibi_is_symmetric_non_causal(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, _ = device_dtype
    bias = ALiBiBuilder(heads=H, causal=False).forward(N, N, device)
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
    sdpa_kernel: SDPAKernel, qkv: QKV
) -> None:
    out = sdpa_kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_with_bias(
    sdpa_kernel: SDPAKernel, qkv: QKV
) -> None:
    device, dtype = qkv.q.device, qkv.q.dtype
    bias = torch.zeros(H, N, N, device=device, dtype=dtype)
    out = sdpa_kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=bias, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_causal(
    sdpa_kernel: SDPAKernel, qkv: QKV
) -> None:
    out = sdpa_kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=None, is_causal=True)
    assert out.shape == (B, H, N, DH)


# --- windowed kernel ---


def test_windowed_kernel_fallback_shape(qkv: QKV) -> None:
    kernel = WindowedSDPAKernel(window_size=64)  # window > N → fallback
    out = kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_active_shape(qkv: QKV) -> None:
    kernel = WindowedSDPAKernel(window_size=4)  # window < N → mask path
    out = kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_causal_shape(qkv: QKV) -> None:
    kernel = WindowedSDPAKernel(window_size=4, causal=True)
    out = kernel(qkv.q, qkv.k, qkv.v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_windowed_kernel_causal_masks_future(
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    kernel = WindowedSDPAKernel(window_size=4, causal=True)
    q = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    k = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
    v = torch.eye(8, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(1, 1, 8, 8)[..., :4]
    q[0, 0, 0] = 1.0
    out = kernel(q, k, v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (1, 1, 8, 4)


# --- varlen SDPA kernel ---


def test_varlen_kernel_shape(
    varlen_kernel: VarlenSDPAKernel, varlen_batch: VarlenBatch
) -> None:
    vb = varlen_batch
    out = varlen_kernel(vb.q, vb.k, vb.v, cu_seqlens=vb.cu_seqlens, max_seqlen=vb.max_seqlen)
    assert out.shape == vb.q.shape


def test_varlen_kernel_causal_shape(
    varlen_causal_kernel: VarlenSDPAKernel, varlen_batch: VarlenBatch
) -> None:
    vb = varlen_batch
    out = varlen_causal_kernel(vb.q, vb.k, vb.v, cu_seqlens=vb.cu_seqlens, max_seqlen=vb.max_seqlen)
    assert out.shape == vb.q.shape


# --- varlen windowed kernel ---


def test_varlen_windowed_kernel_shape(varlen_batch: VarlenBatch) -> None:
    kernel = VarlenWindowedSDPAKernel(window_size=4)
    vb = varlen_batch
    out = kernel(vb.q, vb.k, vb.v, cu_seqlens=vb.cu_seqlens, max_seqlen=vb.max_seqlen)
    assert out.shape == vb.q.shape


def test_varlen_windowed_kernel_causal_shape(varlen_batch: VarlenBatch) -> None:
    kernel = VarlenWindowedSDPAKernel(window_size=3, causal=True)
    vb = varlen_batch
    out = kernel(vb.q, vb.k, vb.v, cu_seqlens=vb.cu_seqlens, max_seqlen=vb.max_seqlen)
    assert out.shape == vb.q.shape
