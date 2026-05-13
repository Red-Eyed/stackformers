from __future__ import annotations

import pytest
import torch

from tests.conftest import atol
from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.positional.rope2d import RotaryEmbedding2D

B, H, N, DH = 2, 4, 8, 32


@pytest.fixture
def qk(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
    )


@pytest.fixture
def rope1d(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding1D:
    device, dtype = device_dtype
    return RotaryEmbedding1D(dim_head=DH).to(device=device, dtype=dtype)


@pytest.fixture
def rope2d(device_dtype: tuple[torch.device, torch.dtype]) -> RotaryEmbedding2D:
    device, dtype = device_dtype
    return RotaryEmbedding2D(dim_head=DH).to(device=device, dtype=dtype)


def test_no_pos_encoding_passthrough(
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_out, k_out = NoPosEncoding()(q, k)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_rope1d_output_shape(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
) -> None:
    q, k = qk
    q_out, k_out = rope1d(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope1d_cross_attn_different_lengths(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 6, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 12, DH, device=device, dtype=dtype)
    q_out, k_out = rope1d(q, k)
    assert q_out.shape == (B, H, 6, DH)
    assert k_out.shape == (B, H, 12, DH)


def test_rope1d_different_positions_produce_different_output(
    rope1d: RotaryEmbedding1D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    k = torch.ones(1, 1, 4, DH, device=device, dtype=dtype)
    q_out, _ = rope1d(q, k)
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_rope1d_preserves_norms(
    rope1d: RotaryEmbedding1D,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    _, dtype = device_dtype
    q, k = qk
    q_out, k_out = rope1d(q, k)
    tol = atol(dtype)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=tol)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=tol)


def test_rope2d_output_shape(
    rope2d: RotaryEmbedding2D,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, N, DH, device=device, dtype=dtype)
    row_ids = torch.arange(N, dtype=dtype, device=device)
    col_ids = torch.arange(N, dtype=dtype, device=device)
    q_out, k_out = rope2d(q, k, row_ids, col_ids)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
