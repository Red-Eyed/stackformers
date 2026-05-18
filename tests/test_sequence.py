from __future__ import annotations

import pytest
import torch

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
