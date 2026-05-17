from __future__ import annotations

import pytest
import torch

from stackformers.positional.config import LearnedPosEncodingConfig
from stackformers.positional.learned import LearnedPosEncoding

B, H, N, DH = 2, 4, 8, 32
MAX_LEN = 64


@pytest.fixture
def config() -> LearnedPosEncodingConfig:
    return LearnedPosEncodingConfig(dim_head=DH, max_seq_len=MAX_LEN)


@pytest.fixture
def enc(
    config: LearnedPosEncodingConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> LearnedPosEncoding:
    device, dtype = device_dtype
    return LearnedPosEncoding(config).to(device=device, dtype=dtype)


@pytest.fixture
def qk(device_dtype: tuple[torch.device, torch.dtype]) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = device_dtype
    return (
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
        torch.randn(B, H, N, DH, device=device, dtype=dtype),
    )


def _positions(b: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sequential 1-D positions as (b, n, 1)."""
    return torch.arange(n, device=device, dtype=dtype).view(1, n, 1).expand(b, -1, -1).clone()


def test_learned_pos_output_shape(
    enc: LearnedPosEncoding,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    pos = _positions(B, N, device, dtype)
    q_out, k_out = enc.forward_padded(q, k, pos, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_learned_pos_different_positions_differ(
    enc: LearnedPosEncoding,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Tokens at different positions receive different offsets."""
    device, dtype = device_dtype
    q = torch.ones(1, H, 2, DH, device=device, dtype=dtype)
    k = torch.ones(1, H, 2, DH, device=device, dtype=dtype)
    pos = _positions(1, 2, device, dtype)
    q_out, _ = enc.forward_padded(q, k, pos, pos)
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_learned_pos_same_position_same_offset(
    enc: LearnedPosEncoding,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Two sequences at the same position index receive the same embedding offset."""
    device, dtype = device_dtype
    q_row = torch.randn(1, H, 1, DH, device=device, dtype=dtype)
    q = q_row.expand(2, -1, -1, -1).clone()
    k = torch.zeros(2, H, 1, DH, device=device, dtype=dtype)
    pos = torch.full((2, 1, 1), 5.0, device=device, dtype=dtype)
    q_out, _ = enc.forward_padded(q, k, pos, pos)
    assert torch.allclose(q_out[0], q_out[1])


def test_learned_pos_cross_attn_different_lengths(
    enc: LearnedPosEncoding,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 4, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 10, DH, device=device, dtype=dtype)
    q_pos = _positions(B, 4, device, dtype)
    k_pos = _positions(B, 10, device, dtype)
    q_out, k_out = enc.forward_padded(q, k, q_pos, k_pos)
    assert q_out.shape == (B, H, 4, DH)
    assert k_out.shape == (B, H, 10, DH)


def test_learned_pos_gradients_flow(device: torch.device) -> None:
    enc = LearnedPosEncoding(LearnedPosEncodingConfig(dim_head=DH, max_seq_len=MAX_LEN)).to(device)
    q = torch.randn(B, H, N, DH, device=device, requires_grad=True)
    k = torch.randn(B, H, N, DH, device=device)
    pos = _positions(B, N, device, torch.float32)
    q_out, _ = enc.forward_padded(q, k, pos, pos)
    q_out.sum().backward()
    assert q.grad is not None
    assert enc.emb.weight.grad is not None
