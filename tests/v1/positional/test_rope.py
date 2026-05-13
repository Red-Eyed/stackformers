from __future__ import annotations

import torch

from stackformers.v1.positional.none import NoPosEncoding
from stackformers.v1.positional.rope1d import RotaryEmbedding1D
from stackformers.v1.positional.rope2d import RotaryEmbedding2D

B, H, N, DH = 2, 4, 8, 32


def _qk() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(B, H, N, DH), torch.randn(B, H, N, DH)


def test_no_pos_encoding_passthrough() -> None:
    rope = NoPosEncoding()
    q, k = _qk()
    q_out, k_out = rope(q, k)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_rope1d_output_shape() -> None:
    rope = RotaryEmbedding1D(dim_head=DH)
    q, k = _qk()
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope1d_cross_attn_different_lengths() -> None:
    rope = RotaryEmbedding1D(dim_head=DH)
    q = torch.randn(B, H, 6, DH)
    k = torch.randn(B, H, 12, DH)
    q_out, k_out = rope(q, k)
    assert q_out.shape == (B, H, 6, DH)
    assert k_out.shape == (B, H, 12, DH)


def test_rope1d_different_positions_produce_different_output() -> None:
    rope = RotaryEmbedding1D(dim_head=DH)
    q = torch.ones(1, 1, 4, DH)
    k = torch.ones(1, 1, 4, DH)
    q_out, _ = rope(q, k)
    assert not torch.allclose(q_out[:, :, 0], q_out[:, :, 1])


def test_rope1d_preserves_norms() -> None:
    """RoPE is a rotation, so it must preserve the L2 norm of each head vector."""
    rope = RotaryEmbedding1D(dim_head=DH)
    q, k = _qk()
    q_out, k_out = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5)


def test_rope2d_output_shape() -> None:
    dh4 = 32  # must be divisible by 4
    rope = RotaryEmbedding2D(dim_head=dh4)
    q = torch.randn(B, H, N, dh4)
    k = torch.randn(B, H, N, dh4)
    row_ids = torch.arange(N, dtype=torch.float)
    col_ids = torch.arange(N, dtype=torch.float)
    q_out, k_out = rope(q, k, row_ids, col_ids)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
