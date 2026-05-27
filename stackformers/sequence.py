from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Bool, Float, Int
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


class PaddedInput(NamedTuple):
    x: Float[Tensor, "b n d"]
    mask: Bool[Tensor, "b n"]
    abs_positions: Float[Tensor, "b n c"]  # c=1 for 1-D (text), c=2 for grids, c=3 for 3-D


class PackedInput(NamedTuple):
    x: Float[Tensor, "nt d"]
    cu_seqlens: Int[Tensor, "bp1"]
    max_seqlen: int
    abs_positions: Float[Tensor, "nt c"]  # c matches PaddedInput convention


SequenceInput = PaddedInput | PackedInput


def to_seq_info(inp: SequenceInput) -> SequenceInfo:
    match inp:
        case PaddedInput(mask=mask):
            return PaddedSequence(mask=mask)
        case PackedInput(cu_seqlens=cu, max_seqlen=ms):
            return PackedSequence(cu_seqlens=cu, max_seqlen=ms)


def make_padded_input(x: Tensor, mask: Bool[Tensor, "b n"]) -> PaddedInput:
    """Build PaddedInput with sequential 1-D positions (shape b n 1)."""
    n = x.shape[1]
    pos = torch.arange(n, device=x.device, dtype=x.dtype)
    positions = pos.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, -1)  # b n 1
    return PaddedInput(x=x, mask=mask, abs_positions=positions)


def make_packed_input(x: Tensor, cu_seqlens: Int[Tensor, "bp1"], max_seqlen: int) -> PackedInput:
    """Build PackedInput with per-token sequential 1-D positions (shape nt 1).

    Delegates to :func:`position_ids_from_packed`, which is ``torch.export``-compatible.
    """
    seq = PackedSequence(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    positions = position_ids_from_packed(seq).to(x.dtype).unsqueeze(-1)  # nt 1
    return PackedInput(x=x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, abs_positions=positions)


def padded_to_key_padding_mask(seq: PaddedSequence) -> Bool[Tensor, "b n"]:
    """Return True where tokens are valid (mirrors seq.mask)."""
    return seq.mask


def packed_batch_size(seq: PackedSequence) -> int:
    return int(seq.cu_seqlens.shape[0]) - 1


def make_padded(mask: Bool[Tensor, "b n"]) -> PaddedSequence:
    return PaddedSequence(mask=mask)


def make_packed(cu_seqlens: Int[Tensor, "bp1"], max_seqlen: int) -> PackedSequence:
    return PackedSequence(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def padded_to_packed(inp: PaddedInput) -> PackedInput:
    """Remove padding tokens from a padded batch, producing a flat packed tensor.

    Output shape is data-dependent (nt = mask.sum()); not compatible with torch.export.
    """
    lengths = inp.mask.sum(dim=1)
    cu = lengths_to_cu_seqlens(lengths)
    max_seqlen = int(lengths.max().item())
    return PackedInput(
        x=inp.x[inp.mask],
        cu_seqlens=cu,
        max_seqlen=max_seqlen,
        abs_positions=inp.abs_positions[inp.mask],
    )


def packed_to_padded(inp: PackedInput) -> PaddedInput:
    """Re-pad a packed tensor to (b, max_seqlen, d), filling unused positions with zeros.

    Uses scatter — compatible with torch.export and torch.compile.
    """
    cu = inp.cu_seqlens
    b = cu.shape[0] - 1
    n = inp.max_seqlen
    # Build a (nt,) tensor of batch indices and a (nt,) tensor of within-sequence positions.
    lengths = cu[1:] - cu[:-1]  # (b,)
    batch_idx = torch.repeat_interleave(
        torch.arange(b, device=cu.device, dtype=torch.long), lengths
    )  # (nt,)
    pos_idx = position_ids_from_packed(PackedSequence(cu_seqlens=cu, max_seqlen=n))  # (nt,)
    d, c = inp.x.shape[-1], inp.abs_positions.shape[-1]
    x_out = inp.x.new_zeros(b, n, d)
    x_out[batch_idx, pos_idx] = inp.x
    pos_out = inp.abs_positions.new_zeros(b, n, c)
    pos_out[batch_idx, pos_idx] = inp.abs_positions
    mask = torch.zeros(b, n, dtype=torch.bool, device=inp.x.device)
    mask[batch_idx, pos_idx] = True
    return PaddedInput(x=x_out, mask=mask, abs_positions=pos_out)


def lengths_to_cu_seqlens(lengths: Int[Tensor, "b"]) -> Int[Tensor, "bp1"]:
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, lengths.cumsum(0)])


def position_ids_from_packed(seq: PackedSequence) -> Int[Tensor, "nt"]:
    """Build per-token position indices [0..len_i-1] for each document in the pack.

    The result at flat index i is ``i - cu_seqlens[batch_of_i]``, i.e. the within-document
    position of that token.

    Compatible with ``torch.export`` and ``torch.compile``: no Python-side iteration over
    data-dependent lengths.  The key identity is::

        pos_idx = arange(nt) - cu[batch_idx]

    where ``batch_idx`` is built via ``repeat_interleave``, which is a supported dynamic op.
    """
    cu = seq.cu_seqlens
    b = cu.shape[0] - 1
    lengths = cu[1:] - cu[:-1]  # (b,) — kept as a tensor, never converted to Python list
    batch_idx = torch.repeat_interleave(
        torch.arange(b, device=cu.device, dtype=torch.long), lengths
    )  # (nt,)
    nt = batch_idx.shape[0]
    return torch.arange(nt, device=cu.device, dtype=torch.long) - cu[batch_idx]
