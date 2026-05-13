from __future__ import annotations

import pytest
import torch

from stackformers.sequence import (
    PackedSequence,
    PaddedSequence,
    SequenceInfo,
    lengths_to_cu_seqlens,
    make_packed,
    make_padded,
    packed_batch_size,
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


def test_sequence_info_union() -> None:
    mask = torch.ones(1, 4, dtype=torch.bool)
    padded: SequenceInfo = make_padded(mask)
    assert isinstance(padded, PaddedSequence)

    cu = torch.tensor([0, 4])
    packed: SequenceInfo = make_packed(cu, max_seqlen=4)
    assert isinstance(packed, PackedSequence)
