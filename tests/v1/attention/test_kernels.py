from __future__ import annotations

import torch

from stackformers.v1.attention.bias import ALiBiBuilder, NoBiasBuilder
from stackformers.v1.attention.kernels import SDPAKernel, WindowedSDPAKernel

B, H, N, DH = 2, 4, 16, 32


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(B, H, N, DH),
        torch.randn(B, H, N, DH),
        torch.randn(B, H, N, DH),
    )


# --- bias builders ---


def test_no_bias_builder_returns_none() -> None:
    builder = NoBiasBuilder()
    result = builder.forward(N, N, torch.device("cpu"))
    assert result is None


def test_alibi_output_shape() -> None:
    builder = ALiBiBuilder(heads=H)
    bias = builder.forward(N, N, torch.device("cpu"))
    assert bias is not None
    assert bias.shape == (H, N, N)


def test_alibi_is_symmetric_non_causal() -> None:
    builder = ALiBiBuilder(heads=H, causal=False)
    bias = builder.forward(N, N, torch.device("cpu"))
    assert bias is not None
    assert torch.allclose(bias, bias.transpose(-1, -2))


def test_alibi_slopes_positive_different() -> None:
    builder = ALiBiBuilder(heads=4)
    bias = builder.forward(N, N, torch.device("cpu"))
    assert bias is not None
    assert not torch.allclose(bias[0], bias[1])


def test_alibi_non_power_of_two_heads() -> None:
    builder = ALiBiBuilder(heads=6)
    bias = builder.forward(N, N, torch.device("cpu"))
    assert bias is not None
    assert bias.shape == (6, N, N)


# --- SDPA kernel ---


def test_sdpa_kernel_output_shape() -> None:
    kernel = SDPAKernel()
    q, k, v = _qkv()
    out = kernel(q, k, v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_with_bias() -> None:
    kernel = SDPAKernel()
    q, k, v = _qkv()
    bias = torch.zeros(H, N, N)
    out = kernel(q, k, v, attn_mask=None, attn_bias=bias, is_causal=False)
    assert out.shape == (B, H, N, DH)


def test_sdpa_kernel_causal() -> None:
    kernel = SDPAKernel()
    q, k, v = _qkv()
    out = kernel(q, k, v, attn_mask=None, attn_bias=None, is_causal=True)
    assert out.shape == (B, H, N, DH)


# --- windowed kernel ---


def test_windowed_kernel_fallback_shape() -> None:
    kernel = WindowedSDPAKernel(window_size=64)  # window > N → fallback
    q, k, v = _qkv()
    out = kernel(q, k, v, attn_mask=None, attn_bias=None, is_causal=False)
    assert out.shape == (B, H, N, DH)
