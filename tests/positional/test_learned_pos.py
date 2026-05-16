from __future__ import annotations

import pytest
import torch

from stackformers.positional.config import LearnedPosEncodingConfig
from stackformers.positional.learned import LearnedPosEncoding
from stackformers.sequence import make_padded_input

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


def _padded(n: int, device: torch.device, dtype: torch.dtype) -> object:
    x = torch.zeros(B, n, 1, device=device, dtype=dtype)
    mask = torch.ones(B, n, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_learned_pos_output_shape(
    enc: LearnedPosEncoding,
    qk: tuple[torch.Tensor, torch.Tensor],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q, k = qk
    inp = _padded(N, device, dtype)
    q_out, k_out = enc(q, k, inp, inp)  # type: ignore[arg-type]
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
    inp = _padded(2, device, dtype)
    q_out, _ = enc(q, k, inp, inp)  # type: ignore[arg-type]
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_learned_pos_same_position_same_offset(
    enc: LearnedPosEncoding,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Two sequences at the same position index receive the same embedding offset."""
    from stackformers.sequence import PaddedInput

    device, dtype = device_dtype
    # Use identical q for both batch items so (q_out - q) is exact even in fp16.
    q_row = torch.randn(1, H, 1, DH, device=device, dtype=dtype)
    q = q_row.expand(2, -1, -1, -1).clone()
    k = torch.zeros(2, H, 1, DH, device=device, dtype=dtype)
    x = torch.zeros(2, 1, 1, device=device, dtype=dtype)
    mask = torch.ones(2, 1, dtype=torch.bool, device=device)
    pos = torch.full((2, 1, 1), 5.0, device=device, dtype=dtype)
    inp = PaddedInput(x=x, mask=mask, abs_positions=pos)
    q_out, _ = enc(q, k, inp, inp)
    # Both batch items started equal and got the same embedding → still equal.
    assert torch.allclose(q_out[0], q_out[1])


def test_learned_pos_cross_attn_different_lengths(
    enc: LearnedPosEncoding,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    q = torch.randn(B, H, 4, DH, device=device, dtype=dtype)
    k = torch.randn(B, H, 10, DH, device=device, dtype=dtype)
    q_inp = _padded(4, device, dtype)
    k_inp = _padded(10, device, dtype)
    q_out, k_out = enc(q, k, q_inp, k_inp)  # type: ignore[arg-type]
    assert q_out.shape == (B, H, 4, DH)
    assert k_out.shape == (B, H, 10, DH)


def test_learned_pos_gradients_flow(device: torch.device) -> None:
    enc = LearnedPosEncoding(LearnedPosEncodingConfig(dim_head=DH, max_seq_len=MAX_LEN)).to(device)
    q = torch.randn(B, H, N, DH, device=device, requires_grad=True)
    k = torch.randn(B, H, N, DH, device=device)
    inp = _padded(N, device, torch.float32)
    q_out, _ = enc(q, k, inp, inp)  # type: ignore[arg-type]
    q_out.sum().backward()
    assert q.grad is not None
    assert enc.emb.weight.grad is not None
