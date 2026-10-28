"""Padded and packed tensor values, pure conversions, and legacy layout adapters."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple, assert_never

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils._pytree import register_pytree_node
from typing_extensions import deprecated

from stackformers._result import unwrap_or_raise
from stackformers.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from jaxtyping import Bool, Float, Int
    from torch.utils._pytree import KeyEntry

# NamedTuple instead of dataclass: PyTorch's pytree system handles tuples natively,
# so torch.export and torch.compile see the tensor fields without any registration.
# Dataclasses require explicit pytree.register_dataclass() to achieve the same.


@deprecated("Use PaddedTensor with values and validity instead.", category=None)
class PaddedSequence(NamedTuple):
    """Batch of padded sequences; mask marks valid (non-padding) positions."""

    mask: Bool[Tensor, "b n"]  # True = valid token


@deprecated("Use PackedTensor with values and boundaries instead.", category=None)
class PackedSequence(NamedTuple):
    """Variable-length sequences packed into a single flat tensor.

    cu_seqlens[i] is the cumulative token count up to (not including) sequence i.
    Follows the FlashAttention convention: length[i] = cu_seqlens[i+1] - cu_seqlens[i].
    """

    cu_seqlens: Int[Tensor, "bp1"]  # shape (batch + 1,)
    max_seqlen: int


SequenceInfo = PaddedSequence | PackedSequence


class AllValid:
    """Stateless assertion that every dense position is valid; never inferred from values."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Keep dense input representations readable without instance addresses."""
        return "AllValid()"


def _flatten_all_valid(marker: AllValid) -> tuple[list[object], None]:
    """Expose the marker as a static pytree node with no graph input tensors."""
    return [], None


def _flatten_all_valid_keys(marker: AllValid) -> tuple[list[tuple[KeyEntry, object]], None]:
    """Expose no keyed leaves for the stateless marker."""
    return [], None


def _unflatten_all_valid(values: Iterable[object], context: None) -> AllValid:
    """Restore the value-free validity assertion while rebuilding an input tree."""
    return AllValid()


# A zero-leaf node keeps the assertion static across compile, export and shape conversion.
register_pytree_node(
    AllValid,
    _flatten_all_valid,
    _unflatten_all_valid,
    serialized_type_name="stackformers.AllValid",
    flatten_with_keys_fn=_flatten_all_valid_keys,
)


class PaddedTensor(NamedTuple):
    """Dense values, explicit validity, and coordinates; tensor storage is borrowed.

    Direct construction trusts metadata. Factory methods validate external lists.
    Methods never mutate inputs, but views and identity conversions may share storage.
    """

    x: Float[Tensor, "b n d"]
    mask: Bool[Tensor, "b n"] | AllValid
    abs_positions: Float[Tensor, "b n c"]

    @classmethod
    def from_list(
        cls,
        sequences: Sequence[Float[Tensor, "n d"]],
        *,
        abs_positions: Sequence[Float[Tensor, "n c"]] | None = None,
        padding_value: float = 0.0,
    ) -> PaddedTensor:
        """Right-pad a nonempty list, generating local positions unless supplied.

        Zero-length items are supported. Raise ValueError for incompatible shapes,
        devices, dtypes, or position lists. Values retain their autograd history.
        """
        return PackedTensor.from_list(sequences, abs_positions=abs_positions).to_padded(
            padding_value=padding_value
        )

    @classmethod
    def from_nested(cls, tensor: Tensor) -> PaddedTensor:
        """Pad a rank-three nested tensor; generate sequential positions.

        Raise ValueError for a dense tensor or unsupported rank. Custom coordinates
        are not carried by nested tensors; use from_list to supply them separately.
        """
        return PackedTensor.from_nested(tensor).to_padded()

    def to_padded(self) -> PaddedTensor:
        """Return this record unchanged, sharing its storage."""
        return self

    def to_packed(self) -> PackedTensor:
        """Gather valid tokens and coordinates in row order; eager, data-dependent."""
        return _padded_to_packed(self)

    def to_list(self) -> list[Float[Tensor, "n d"]]:
        """Return valid values per row, omitting coordinates; retain gradients.

        AllValid rows are views; masked rows are gathered copies. This eager adapter
        may synchronize device metadata and is not intended for export.
        """
        match self.mask:
            case AllValid():
                return list(self.x.unbind(0))
            case Tensor() as mask:
                return [
                    values[valid]
                    for values, valid in zip(self.x.unbind(), mask.unbind(), strict=True)
                ]

    def to_nested(self) -> Tensor:
        """Return valid values as a jagged nested tensor, omitting coordinates.

        Packing allocates storage for masked inputs. Requires PyTorch jagged support.
        """
        return self.to_packed().to_nested()


class PackedTensor(NamedTuple):
    """Flat token values, cumulative boundaries, and coordinates; storage is borrowed.

    Boundaries start at zero and end at the token count; max_seqlen bounds all
    lengths. Direct construction trusts metadata. Methods do not mutate inputs.
    """

    x: Float[Tensor, "nt d"]
    cu_seqlens: Int[Tensor, "bp1"]
    max_seqlen: int
    abs_positions: Float[Tensor, "nt c"]

    @classmethod
    def from_list(
        cls,
        sequences: Sequence[Float[Tensor, "n d"]],
        *,
        abs_positions: Sequence[Float[Tensor, "n c"]] | None = None,
    ) -> PackedTensor:
        """Concatenate a nonempty list without detaching gradients or moving values.

        Zero-length items are supported. Raise ValueError for incompatible shapes,
        devices, dtypes, or position lists. Positions default to local float32 indices.
        """
        unwrap_or_raise(_validate_sequences(sequences))
        lengths = torch.tensor(
            [item.shape[0] for item in sequences], device=sequences[0].device, dtype=torch.int32
        )
        cu = lengths_to_cu_seqlens(lengths).to(torch.int32)
        x = torch.cat(tuple(sequences))
        maximum = max(item.shape[0] for item in sequences)
        if abs_positions is None:
            positions = torch.cat(
                tuple(
                    torch.arange(item.shape[0], device=x.device, dtype=torch.float32)[:, None]
                    for item in sequences
                )
            )
        else:
            unwrap_or_raise(_validate_positions(sequences, abs_positions))
            positions = torch.cat(tuple(abs_positions))
        return cls(x=x, cu_seqlens=cu, max_seqlen=maximum, abs_positions=positions)

    @classmethod
    def from_nested(cls, tensor: Tensor) -> PackedTensor:
        """Pack rank-three nested values, preserving gradients and generating positions.

        Supports strided and jagged nested layouts, including jagged holes. Raise
        ValueError for dense tensors, unsupported rank, or an empty batch.
        """
        if not tensor.is_nested or tensor.ndim != 3:
            raise ValueError("expected a rank-three nested tensor")
        return cls.from_list(tensor.unbind(0))

    def to_packed(self) -> PackedTensor:
        """Return this record unchanged, sharing its storage."""
        return self

    def to_padded(self, *, padding_value: float = 0.0) -> PaddedTensor:
        """Right-pad to max_seqlen; preserve coordinates and fill unused ones with zero.

        Values use padding_value; valid tokens retain gradients. The tensor-only
        scatter supports compile/export with dynamic output shapes enabled.
        """
        return _packed_to_padded(self, padding_value=padding_value)

    def to_list(self) -> list[Float[Tensor, "n d"]]:
        """Return per-sequence views of values, omitting coordinates; retain gradients.

        Reading boundaries synchronizes GPU metadata. This is an eager adapter.
        """
        lengths = self.cu_seqlens.diff().tolist()
        return list(self.x.split(lengths, dim=0))

    def to_nested(self) -> Tensor:
        """Return a jagged view sharing values, omitting coordinates; retain gradients.

        Requires PyTorch jagged support. Offsets are normalized to int64 on device.
        """
        return torch.nested.nested_tensor_from_jagged(
            self.x, offsets=self.cu_seqlens.to(torch.int64), max_seqlen=self.max_seqlen
        )


# Keep aliases visible to type checkers; runtime lookups warn without wrapping the types.
if TYPE_CHECKING:
    PaddedInput = PaddedTensor
    PackedInput = PackedTensor


def __getattr__(name: str) -> type[PaddedTensor] | type[PackedTensor]:
    """Resolve deprecated input aliases while preserving constructor and match identity."""
    match name:
        case "PaddedInput":
            replacement = PaddedTensor
        case "PackedInput":
            replacement = PackedTensor
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated; use {replacement.__name__} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement


SequenceInput = PaddedTensor | PackedTensor


@deprecated("Use the tensor record directly.", category=None)
def to_seq_info(inp: SequenceInput) -> SequenceInfo:
    """Return legacy metadata, materializing an all-true mask only for all-valid records."""
    match inp:
        case PaddedTensor():
            return PaddedSequence(mask=materialize_padding_mask(inp))
        case PackedTensor(cu_seqlens=cu, max_seqlen=ms):
            return PackedSequence(cu_seqlens=cu, max_seqlen=ms)


@deprecated("Use PaddedTensor with explicit positions or PaddedTensor.from_list.", category=None)
def make_padded_input(x: Tensor, mask: Bool[Tensor, "b n"] | AllValid) -> PaddedTensor:
    """Build dense input with required validity and sequential 1-D positions (b n 1)."""
    n = x.shape[1]
    pos = torch.arange(n, device=x.device, dtype=torch.float32)
    positions = repeat(pos, "n -> b n 1", b=x.shape[0])  # b n 1
    return PaddedTensor(x=x, mask=mask, abs_positions=positions)


@deprecated("Use PackedTensor with explicit positions or PackedTensor.from_list.", category=None)
def make_packed_input(x: Tensor, cu_seqlens: Int[Tensor, "bp1"], max_seqlen: int) -> PackedTensor:
    """Build PackedTensor with per-token sequential 1-D positions (shape nt 1).

    Generates local positions through the shared export-compatible tensor operation.
    """
    positions = _position_ids(cu_seqlens).to(torch.float32)
    positions = rearrange(positions, "nt -> nt 1")  # nt 1
    return PackedTensor(x=x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, abs_positions=positions)


@deprecated("Use materialize_padding_mask on PaddedTensor.", category=None)
def padded_to_key_padding_mask(seq: PaddedSequence) -> Bool[Tensor, "b n"]:
    """Return True where tokens are valid (mirrors seq.mask)."""
    return seq.mask


@deprecated("Use tensor.cu_seqlens.shape[0] - 1.", category=None)
def packed_batch_size(seq: PackedTensor | PackedSequence) -> int:
    """Return the document count without specializing symbolic tensor dimensions."""
    return seq.cu_seqlens.shape[0] - 1


@deprecated("Use PaddedTensor.", category=None)
def make_padded(mask: Bool[Tensor, "b n"]) -> PaddedSequence:
    """Deprecated adapter for callers retaining mask-only metadata."""
    return PaddedSequence(mask=mask)


@deprecated("Use PackedTensor.", category=None)
def make_packed(cu_seqlens: Int[Tensor, "bp1"], max_seqlen: int) -> PackedSequence:
    """Deprecated adapter for callers retaining boundary-only metadata."""
    return PackedSequence(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def _padded_to_packed(inp: PaddedTensor) -> PackedTensor:
    """Remove padding tokens from a padded batch, producing a flat packed tensor.

    Output shape is data-dependent (nt = mask.sum()); not compatible with torch.export.
    """
    match inp.mask:
        case AllValid():
            b, n, _ = inp.x.shape
            cu = torch.arange(b + 1, device=inp.x.device, dtype=torch.int64) * n
            return PackedTensor(
                x=rearrange(inp.x, "b n d -> (b n) d"),
                cu_seqlens=cu,
                max_seqlen=n,
                abs_positions=rearrange(inp.abs_positions, "b n c -> (b n) c"),
            )
        case Tensor() as mask:
            pass
    lengths = mask.sum(dim=1)
    cu = lengths_to_cu_seqlens(lengths)
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    return PackedTensor(
        x=inp.x[mask],
        cu_seqlens=cu,
        max_seqlen=max_seqlen,
        abs_positions=inp.abs_positions[mask],
    )


def _packed_to_padded(inp: PackedTensor, *, padding_value: float = 0.0) -> PaddedTensor:
    """Scatter packed values and coordinates, filling unused values with padding_value.

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
    pos_idx = _position_ids(cu)  # (nt,)
    d, c = inp.x.shape[-1], inp.abs_positions.shape[-1]
    x_out = inp.x.new_full((b, n, d), padding_value)
    x_out[batch_idx, pos_idx] = inp.x
    pos_out = inp.abs_positions.new_zeros(b, n, c)
    pos_out[batch_idx, pos_idx] = inp.abs_positions
    mask = torch.zeros(b, n, dtype=torch.bool, device=inp.x.device)
    mask[batch_idx, pos_idx] = True
    return PaddedTensor(x=x_out, mask=mask, abs_positions=pos_out)


def lengths_to_cu_seqlens(lengths: Int[Tensor, "b"]) -> Int[Tensor, "bp1"]:
    """Prepend zero to cumulative document lengths on the original device."""
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, lengths.cumsum(0)])


def position_ids_from_packed(seq: PackedTensor | PackedSequence) -> Int[Tensor, "nt"]:
    """Build per-token position indices [0..len_i-1] for each document in the pack.

    The result at flat index i is ``i - cu_seqlens[batch_of_i]``, i.e. the within-document
    position of that token.

    Compatible with ``torch.export`` and ``torch.compile``: no Python-side iteration over
    data-dependent lengths.  The key identity is::

        pos_idx = arange(nt) - cu[batch_idx]

    where ``batch_idx`` is built via ``repeat_interleave``, which is a supported dynamic op.
    """
    return _position_ids(seq.cu_seqlens)


def _position_ids(cu: Int[Tensor, "bp1"]) -> Int[Tensor, "nt"]:
    """Compute local token indices from cumulative boundaries without Python iteration."""
    b = cu.shape[0] - 1
    lengths = cu[1:] - cu[:-1]  # (b,) — kept as a tensor, never converted to Python list
    batch_idx = torch.repeat_interleave(
        torch.arange(b, device=cu.device, dtype=torch.long), lengths
    )  # (nt,)
    nt = batch_idx.shape[0]
    return torch.arange(nt, device=cu.device, dtype=torch.long) - cu[batch_idx]


def materialize_padding_mask(input: PaddedTensor) -> Bool[Tensor, "b n"]:
    """Return the tensor mask unchanged, allocating ones only for explicit AllValid.

    Use this at adapters requiring a tensor mask. Attention dispatches on the
    marker directly and does not materialize it on the all-valid inference path.
    """
    match input.mask:
        case Tensor() as mask:
            return mask
        case AllValid():
            return torch.ones(input.x.shape[:2], dtype=torch.bool, device=input.x.device)
        case _:
            raise TypeError("mask must be a tensor or explicit AllValid()")


@deprecated("Use tensor.to_packed().", category=None)
def padded_to_packed(inp: PaddedTensor) -> PackedTensor:
    """Deprecated compatibility wrapper; gather valid values and coordinates."""
    return inp.to_packed()


@deprecated("Use tensor.to_padded().", category=None)
def packed_to_padded(inp: PackedTensor) -> PaddedTensor:
    """Deprecated compatibility wrapper; pad values and coordinates with zeros."""
    return inp.to_padded()


def _validate_sequences(sequences: Sequence[Tensor]) -> Result[None, ValueError]:
    """Admit a nonempty batch with a shared feature width, dtype, and device."""
    if not sequences:
        return Err(ValueError("expected at least one sequence"))
    first = sequences[0]
    for item in sequences:
        if item.ndim != 2 or item.shape[1:] != first.shape[1:]:
            return Err(ValueError("sequences must have shape (n, d) with matching feature widths"))
        if item.device != first.device or item.dtype != first.dtype:
            return Err(ValueError("sequences must share dtype and device"))
    return Ok(None)


def _validate_positions(
    sequences: Sequence[Tensor], positions: Sequence[Tensor]
) -> Result[None, ValueError]:
    """Admit one coordinate array per sequence on the same device."""
    if len(positions) != len(sequences):
        return Err(ValueError("positions must contain one tensor per sequence"))
    outcome = _validate_sequences(positions)
    match outcome:
        case Err():
            return outcome
        case Ok():
            pass
        case _:
            assert_never(outcome)
    for values, coordinates in zip(sequences, positions, strict=True):
        if coordinates.shape[0] != values.shape[0] or coordinates.device != values.device:
            return Err(ValueError("positions must match each sequence length and device"))
    return Ok(None)


__all__ = [
    "AllValid",
    "PackedTensor",
    "PaddedTensor",
    "PackedInput",
    "PaddedInput",
    "PackedSequence",
    "PaddedSequence",
    "SequenceInput",
    "SequenceInfo",
    "to_seq_info",
    "make_padded_input",
    "make_packed_input",
    "make_padded",
    "make_packed",
    "padded_to_key_padding_mask",
    "packed_batch_size",
    "padded_to_packed",
    "packed_to_padded",
    "lengths_to_cu_seqlens",
    "position_ids_from_packed",
    "materialize_padding_mask",
]
