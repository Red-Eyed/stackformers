from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Bool, Int
from torch import Tensor

# NamedTuple instead of dataclass: PyTorch's pytree system handles tuples natively,
# so torch.export and torch.compile see the tensor fields without any registration.
# Dataclasses require explicit pytree.register_dataclass() to achieve the same.


class PaddedSequence(NamedTuple):
    """Batch of padded sequences; mask marks valid (non-padding) positions."""

    mask: Bool[Tensor, "b n"]  # True = valid token


class PackedSequence(NamedTuple):
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


def position_ids_from_packed(seq: PackedSequence) -> Int[Tensor, "nt"]:
    """Build per-token position indices [0..len_i-1] for each document in the pack."""
    cu = seq.cu_seqlens
    lengths = (cu[1:] - cu[:-1]).tolist()
    return torch.cat([torch.arange(int(n), device=cu.device, dtype=torch.long) for n in lengths])
