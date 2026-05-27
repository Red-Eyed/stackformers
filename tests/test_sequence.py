from __future__ import annotations

import pytest
import torch
import torch.export
import torch.nn as nn

from stackformers.sequence import (
    PackedInput,
    PackedSequence,
    PaddedInput,
    PaddedSequence,
    SequenceInfo,
    lengths_to_cu_seqlens,
    make_packed,
    make_packed_input,
    make_padded,
    make_padded_input,
    packed_batch_size,
    packed_to_padded,
    padded_to_packed,
    position_ids_from_packed,
)


def test_padded_sequence_mask_shape() -> None:
    mask = torch.ones(2, 8, dtype=torch.bool)
    seq = PaddedSequence(mask=mask)
    assert seq.mask.shape == (2, 8)


def test_padded_sequence_frozen() -> None:
    mask = torch.ones(2, 8, dtype=torch.bool)
    seq = PaddedSequence(mask=mask)
    with pytest.raises(Exception):
        seq.mask = torch.zeros(2, 8, dtype=torch.bool)  # type: ignore[misc]


def test_packed_sequence_fields() -> None:
    cu = torch.tensor([0, 4, 9])
    seq = PackedSequence(cu_seqlens=cu, max_seqlen=5)
    assert seq.max_seqlen == 5
    assert seq.cu_seqlens.shape == (3,)


def test_packed_batch_size() -> None:
    cu = torch.tensor([0, 4, 9, 15])
    seq = PackedSequence(cu_seqlens=cu, max_seqlen=6)
    assert packed_batch_size(seq) == 3


def test_lengths_to_cu_seqlens() -> None:
    lengths = torch.tensor([4, 5, 6])
    cu = lengths_to_cu_seqlens(lengths)
    assert cu.tolist() == [0, 4, 9, 15]


@pytest.fixture
def padded_input_ragged() -> PaddedInput:
    # Two sequences of length 3 and 2, padded to length 4.
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return make_padded_input(x, mask)


@pytest.fixture
def packed_input_ragged() -> PackedInput:
    # Two sequences: length 3 and 2 (nt=5).
    x = torch.randn(5, 8)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    return make_packed_input(x, cu, max_seqlen=3)


def test_padded_to_packed_to_padded_roundtrip(padded_input_ragged: PaddedInput) -> None:
    inp = padded_input_ragged
    out = packed_to_padded(padded_to_packed(inp))
    # out is padded to max_seqlen, not the original n — compare valid tokens only.
    assert torch.equal(out.x[out.mask], inp.x[inp.mask])
    assert torch.equal(out.abs_positions[out.mask], inp.abs_positions[inp.mask])


def test_packed_to_padded_to_packed_roundtrip(packed_input_ragged: PackedInput) -> None:
    inp = packed_input_ragged
    out = padded_to_packed(packed_to_padded(inp))
    assert torch.equal(out.x, inp.x)
    assert torch.equal(out.cu_seqlens, inp.cu_seqlens)
    assert out.max_seqlen == inp.max_seqlen
    assert torch.equal(out.abs_positions, inp.abs_positions)


def test_sequence_info_union() -> None:
    mask = torch.ones(1, 4, dtype=torch.bool)
    padded: SequenceInfo = make_padded(mask)
    assert isinstance(padded, PaddedSequence)

    cu = torch.tensor([0, 4])
    packed: SequenceInfo = make_packed(cu, max_seqlen=4)
    assert isinstance(packed, PackedSequence)


# --- position_ids_from_packed ---


def test_position_ids_from_packed_values() -> None:
    """Per-token position indices must restart at 0 for each document."""
    cu = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    ids = position_ids_from_packed(PackedSequence(cu_seqlens=cu, max_seqlen=4))
    assert ids.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]


def test_position_ids_from_packed_single_doc() -> None:
    """Single document: indices are simply 0..L-1."""
    cu = torch.tensor([0, 5], dtype=torch.long)
    ids = position_ids_from_packed(PackedSequence(cu_seqlens=cu, max_seqlen=5))
    assert ids.tolist() == [0, 1, 2, 3, 4]


def test_make_packed_input_positions() -> None:
    """abs_positions produced by make_packed_input must equal per-document arange."""
    cu = torch.tensor([0, 3, 5], dtype=torch.long)
    x = torch.randn(5, 8)
    inp = make_packed_input(x, cu, max_seqlen=3)
    expected = torch.tensor([0, 1, 2, 0, 1], dtype=torch.float32)
    assert torch.equal(inp.abs_positions.squeeze(-1), expected)


# --- torch.export compatibility ---


class _PosIdsWrapper(nn.Module):
    """Thin nn.Module wrapper so torch.export can trace position_ids_from_packed.

    max_seqlen is not used in the computation; pass any integer constant.
    """

    def forward(self, cu: torch.Tensor) -> torch.Tensor:
        return position_ids_from_packed(PackedSequence(cu_seqlens=cu, max_seqlen=0))


def _export_pos_ids(max_batch: int = 64) -> torch.export.ExportedProgram:
    """Export _PosIdsWrapper with a dynamic bp1 (batch + 1) dimension.

    The total token count nt = cu[-1] is data-dependent (an unbacked symbol) — it varies
    with both the number of sequences (bp1) and the individual sequence lengths.
    """
    wrapper = _PosIdsWrapper()
    cu = torch.tensor([0, 3, 5], dtype=torch.long)  # 2 sequences, nt=5
    bp1_dim = torch.export.Dim("bp1", min=3, max=max_batch + 1)
    return torch.export.export(wrapper, (cu,), dynamic_shapes=({0: bp1_dim},))


def test_position_ids_export_succeeds() -> None:
    """position_ids_from_packed must be traceable by torch.export (no .tolist())."""
    assert _export_pos_ids() is not None


def test_position_ids_export_new_batch_and_seqlens() -> None:
    """Exported program runs correctly for a different batch size and different sequence lengths."""
    mod = _export_pos_ids().module()
    # 3 sequences, lengths 4/3/3 → nt=10 (both batch and nt differ from trace)
    cu = torch.tensor([0, 4, 7, 10], dtype=torch.long)
    assert mod(cu).tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]


def test_position_ids_export_same_batch_different_seqlens() -> None:
    """Exported program runs correctly when nt changes but batch size stays the same."""
    mod = _export_pos_ids().module()
    # Same batch=2 as the trace, but different lengths → different nt
    cu = torch.tensor([0, 6, 11], dtype=torch.long)  # lengths 6/5, nt=11
    assert mod(cu).tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
