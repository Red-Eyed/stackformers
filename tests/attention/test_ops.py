"""Tests for stackformers/attention/ops.py.

Covers:
- Correctness of _cu_to_indices (batch_idx and pos_idx values).
- Correctness of _packed_heads_to_padded / _padded_heads_to_packed round-trip.
- torch.export compatibility: both _cu_to_indices and _packed_heads_to_padded must be
  traceable with dynamic batch size *and* dynamic total token count (nt).
"""

from __future__ import annotations

import torch
import torch.export
import torch.nn as nn

from stackformers.attention.ops import (
    _cu_to_indices,
    _packed_heads_to_padded,
    _padded_heads_to_packed,
)

B = 3
H = 2
DH = 8


# --- _cu_to_indices correctness ---


def test_cu_to_indices_batch_idx() -> None:
    """batch_idx[i] must be the index of the document containing flat token i."""
    cu = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    batch_idx, _ = _cu_to_indices(cu, b=3)
    assert batch_idx.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_cu_to_indices_pos_idx() -> None:
    """pos_idx[i] must be the within-document offset (restarting at 0 per document)."""
    cu = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    _, pos_idx = _cu_to_indices(cu, b=3)
    assert pos_idx.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]


def test_cu_to_indices_single_doc() -> None:
    """Single document: batch_idx is all zeros, pos_idx is arange(nt)."""
    cu = torch.tensor([0, 6], dtype=torch.long)
    batch_idx, pos_idx = _cu_to_indices(cu, b=1)
    assert batch_idx.tolist() == [0, 0, 0, 0, 0, 0]
    assert pos_idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_cu_to_indices_equal_lengths() -> None:
    """Uniform-length sequences: indices are fully regular."""
    cu = torch.tensor([0, 4, 8, 12], dtype=torch.long)
    batch_idx, pos_idx = _cu_to_indices(cu, b=3)
    expected_batch = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    expected_pos = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    assert batch_idx.tolist() == expected_batch
    assert pos_idx.tolist() == expected_pos


# --- _packed_heads_to_padded / _padded_heads_to_packed round-trip ---


def test_packed_to_padded_to_packed_roundtrip() -> None:
    """Valid tokens survive a packed→padded→packed scatter/gather round-trip."""
    cu = torch.tensor([0, 3, 5], dtype=torch.long)
    nt = 5
    x = torch.randn(nt, H, DH)
    x_pad, mask = _packed_heads_to_padded(x, cu, b=2, n=3)
    assert x_pad.shape == (2, H, 3, DH)
    assert mask.shape == (2, 3)
    x_back = _padded_heads_to_packed(x_pad, mask)
    assert torch.allclose(x_back, x)


def test_packed_to_padded_padding_is_zero() -> None:
    """Padding positions in the padded tensor must be zero-filled."""
    # Two seqs: lengths 2 and 3; padded to max_seqlen=3 → seq 0 has one padding slot.
    cu = torch.tensor([0, 2, 5], dtype=torch.long)
    x = torch.randn(5, H, DH)
    x_pad, mask = _packed_heads_to_padded(x, cu, b=2, n=3)
    # mask[0, 2] should be False (padding); x_pad[0, :, 2, :] should be zero.
    assert not mask[0, 2].item()
    assert torch.all(x_pad[0, :, 2, :] == 0)


# --- torch.export compatibility ---


class _CuToIndicesWrapper(nn.Module):
    """Wraps _cu_to_indices as an nn.Module for torch.export tracing.

    ``b`` is derived from ``cu.shape[0] - 1`` rather than passed as a plain int
    argument, so it remains a symbolic expression during export and the batch
    dimension stays fully dynamic.
    """

    def forward(self, cu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = cu.shape[0] - 1  # SymInt when traced; avoids specialising the batch size
        return _cu_to_indices(cu, b)


def _export_cu_to_indices(max_batch: int = 64) -> torch.export.ExportedProgram:
    """Export _CuToIndicesWrapper with a dynamic bp1 dimension.

    nt (total tokens) is data-dependent (derived from cu values) and varies
    independently of bp1, exercising the unbacked-symbol path in torch.export.
    """
    wrapper = _CuToIndicesWrapper()
    cu = torch.tensor([0, 3, 5], dtype=torch.long)  # 2 sequences, nt=5
    bp1_dim = torch.export.Dim("bp1", min=3, max=max_batch + 1)
    return torch.export.export(wrapper, (cu,), dynamic_shapes=({0: bp1_dim},))


def test_cu_to_indices_export_succeeds() -> None:
    """_cu_to_indices must be traceable by torch.export (no .tolist())."""
    assert _export_cu_to_indices() is not None


def test_cu_to_indices_export_new_batch_and_seqlens() -> None:
    """Exported _cu_to_indices produces correct indices for a different batch and nt."""
    mod = _export_cu_to_indices().module()
    cu = torch.tensor([0, 4, 7, 10], dtype=torch.long)  # 3 sequences, nt=10
    batch_idx, pos_idx = mod(cu)
    assert batch_idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert pos_idx.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]


def test_cu_to_indices_export_same_batch_different_seqlens() -> None:
    """Exported _cu_to_indices runs when nt changes but batch size stays the same."""
    mod = _export_cu_to_indices().module()
    # Same batch=2 as trace, different lengths → different nt
    cu = torch.tensor([0, 6, 11], dtype=torch.long)  # lengths 6/5, nt=11
    batch_idx, pos_idx = mod(cu)
    assert batch_idx.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert pos_idx.tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
