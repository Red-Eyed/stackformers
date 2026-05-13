from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor


@dataclass(frozen=True)
class PaddedSequence:
    """Batch of padded sequences; mask marks valid (non-padding) positions."""

    mask: Bool[Tensor, "b n"]  # True = valid token


@dataclass(frozen=True)
class PackedSequence:
    """Variable-length sequences packed into a single flat tensor.

    cu_seqlens[i] is the cumulative token count up to (not including) sequence i.
    Follows the FlashAttention convention: length[i] = cu_seqlens[i+1] - cu_seqlens[i].
    """

    cu_seqlens: Int[Tensor, "bp1"]  # shape (batch + 1,)
    max_seqlen: int


SequenceInfo = PaddedSequence | PackedSequence


def padded_to_key_padding_mask(seq: PaddedSequence) -> Bool[Tensor, "b n"]:
    """Return True where tokens are valid (mirrors seq.mask)."""
    return seq.mask


def packed_batch_size(seq: PackedSequence) -> int:
    return int(seq.cu_seqlens.shape[0]) - 1


def make_padded(mask: Bool[Tensor, "b n"]) -> PaddedSequence:
    return PaddedSequence(mask=mask)


def make_packed(cu_seqlens: Int[Tensor, "bp1"], max_seqlen: int) -> PackedSequence:
    return PackedSequence(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def lengths_to_cu_seqlens(lengths: Int[Tensor, "b"]) -> Int[Tensor, "bp1"]:
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, lengths.cumsum(0)])
