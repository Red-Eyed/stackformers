from __future__ import annotations

import pytest
import torch
import torch.export
import torch.nn as nn
from typing_extensions import override

from stackformers.sequence import (
    AllValid,
    PackedInput,
    PackedSequence,
    PackedTensor,
    PaddedInput,
    PaddedSequence,
    PaddedTensor,
    SequenceInfo,
    lengths_to_cu_seqlens,
    make_packed,
    make_packed_input,
    make_padded,
    make_padded_input,
    materialize_padding_mask,
    packed_batch_size,
    packed_to_padded,
    padded_to_packed,
    position_ids_from_packed,
)
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run


def test_padded_sequence_mask_shape() -> None:
    mask = torch.ones(2, 8, dtype=torch.bool)
    seq = PaddedSequence(mask=mask)
    assert seq.mask.shape == (2, 8)


def test_padded_sequence_frozen() -> None:
    mask = torch.ones(2, 8, dtype=torch.bool)
    seq = PaddedSequence(mask=mask)
    with pytest.raises(AttributeError, match="can't set attribute"):
        # NamedTuple immutability must also hold for untyped runtime callers.
        seq.mask = torch.zeros(2, 8, dtype=torch.bool)  # pyrefly: ignore[read-only]


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
    assert torch.equal(out.x[materialize_padding_mask(out)], inp.x[materialize_padding_mask(inp)])
    assert torch.equal(
        out.abs_positions[materialize_padding_mask(out)],
        inp.abs_positions[materialize_padding_mask(inp)],
    )


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

    @override
    def forward(self, cu: torch.Tensor) -> torch.Tensor:
        return position_ids_from_packed(PackedSequence(cu_seqlens=cu, max_seqlen=0))


def _export_pos_ids(
    shape_mode: ExportShapeMode,
    opset_version: int,
    runtime_cu: torch.Tensor | None = None,
    max_batch: int = 64,
) -> torch.Tensor:
    """Export and run _PosIdsWrapper with a concrete or symbolic batch dimension.

    The total token count nt = cu[-1] is data-dependent (an unbacked symbol) — it varies
    with both the number of sequences (bp1) and the individual sequence lengths.
    """
    wrapper = _PosIdsWrapper()
    cu = torch.tensor([0, 3, 5], dtype=torch.long)  # 2 sequences, nt=5
    bp1_dim = torch.export.Dim("bp1", min=3, max=max_batch + 1)
    (positions,) = export_and_run(
        wrapper,
        (cu,),
        shape_mode,
        opset_version,
        dynamic_shapes=({0: bp1_dim},),
        runtime_args=(runtime_cu,) if runtime_cu is not None else None,
    )
    return positions


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_position_ids_export_succeeds(
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Packed position IDs export with concrete and symbolic batch dimensions."""
    assert _export_pos_ids(shape_mode, opset_version).tolist() == [0, 1, 2, 0, 1]


@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_position_ids_export_new_batch_and_seqlens(opset_version: int) -> None:
    """Dynamic exports run with a different batch size and sequence lengths."""
    # 3 sequences, lengths 4/3/3 → nt=10 (both batch and nt differ from trace)
    cu = torch.tensor([0, 4, 7, 10], dtype=torch.long)
    assert _export_pos_ids(ExportShapeMode.DYNAMIC, opset_version, cu).tolist() == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        0,
        1,
        2,
    ]


@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_position_ids_export_same_batch_different_seqlens(opset_version: int) -> None:
    """Dynamic exports run when token count changes but batch size stays fixed."""
    # Same batch=2 as the trace, but different lengths → different nt
    cu = torch.tensor([0, 6, 11], dtype=torch.long)  # lengths 6/5, nt=11
    assert _export_pos_ids(ExportShapeMode.DYNAMIC, opset_version, cu).tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        1,
        2,
        3,
        4,
    ]


@pytest.fixture
def tensor_list(device_dtype: tuple[torch.device, torch.dtype]) -> list[torch.Tensor]:
    """Provide ragged values including an empty document on each supported device/dtype."""
    device, dtype = device_dtype
    return [torch.randn(n, 8, device=device, dtype=dtype) for n in (3, 0, 2)]


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
def test_tensor_list_roundtrip(
    tensor_list: list[torch.Tensor], tensor_type: type[PackedTensor] | type[PaddedTensor]
) -> None:
    """Constructors preserve token order, empty documents, dtype, and device."""
    tensor = tensor_type.from_list(tensor_list)
    for original, restored in zip(tensor_list, tensor.to_list(), strict=True):
        assert torch.equal(original, restored)
        assert restored.dtype == original.dtype
        assert restored.device == original.device
    packed = tensor.to_packed()
    assert packed.cu_seqlens.tolist() == [0, 3, 3, 5]
    assert packed.max_seqlen == 3
    assert packed.abs_positions[:, 0].tolist() == [0, 1, 2, 0, 1]
    assert packed.to_packed() is packed
    padded = tensor.to_padded()
    assert padded.to_padded() is padded


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
def test_tensor_nested_roundtrip(
    tensor_list: list[torch.Tensor], tensor_type: type[PackedTensor] | type[PaddedTensor]
) -> None:
    """Jagged conversions retain values and empty document boundaries."""
    tensor = tensor_type.from_list(tensor_list)
    nested = tensor.to_nested()
    assert nested.layout == torch.jagged
    restored = tensor_type.from_nested(nested)
    for original, actual in zip(tensor_list, restored.to_list(), strict=True):
        assert torch.equal(original, actual)
    if isinstance(tensor, PackedTensor):
        assert nested.values().data_ptr() == tensor.x.data_ptr()


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
def test_tensor_custom_positions(
    tensor_list: list[torch.Tensor], tensor_type: type[PackedTensor] | type[PaddedTensor]
) -> None:
    """Layout changes retain explicit multidimensional coordinates and valid values."""
    positions = [torch.randn(item.shape[0], 2, device=item.device) for item in tensor_list]
    tensor = tensor_type.from_list(tensor_list, abs_positions=positions)
    packed = tensor.to_packed()
    padded = packed.to_padded(padding_value=-7.0)
    assert torch.equal(padded.to_packed().abs_positions, torch.cat(positions))
    assert torch.equal(padded.to_packed().x, torch.cat(tensor_list))
    invalid = ~materialize_padding_mask(padded)
    assert torch.all(padded.x[invalid] == -7.0)
    assert torch.all(padded.abs_positions[invalid] == 0)


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
def test_tensor_all_empty(
    device_dtype: tuple[torch.device, torch.dtype],
    tensor_type: type[PackedTensor] | type[PaddedTensor],
) -> None:
    """A nonempty batch of empty documents keeps batch and feature dimensions."""
    device, dtype = device_dtype
    tensor = tensor_type.from_list([torch.empty(0, 8, device=device, dtype=dtype)] * 2)
    assert tensor.to_packed().cu_seqlens.tolist() == [0, 0, 0]
    assert tensor.to_padded().x.shape == (2, 0, 8)
    assert [item.shape for item in tensor.to_list()] == [(0, 8), (0, 8)]


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
@pytest.mark.parametrize("route", ["list", "nested", "layout"])
def test_tensor_conversion_gradients(
    device: torch.device,
    tensor_type: type[PackedTensor] | type[PaddedTensor],
    route: str,
) -> None:
    """Every conversion route propagates gradients to the original leaf tensors."""
    items = [torch.randn(n, 8, device=device, requires_grad=True) for n in (3, 0, 2)]
    tensor = tensor_type.from_list(items)
    match route:
        case "list":
            values = tensor.to_list()
        case "nested":
            values = PackedTensor.from_nested(tensor.to_nested()).to_list()
        case _:
            values = tensor.to_packed().to_padded().to_list()
    torch.stack([item.square().sum() for item in values]).sum().backward()
    for item in items:
        assert item.grad is not None
        torch.testing.assert_close(item.grad, 2 * item.detach())


@pytest.mark.parametrize("tensor_type", [PackedTensor, PaddedTensor])
@pytest.mark.parametrize("invalid", ["empty", "rank", "width", "dtype"])
def test_tensor_rejects_invalid_list(
    tensor_type: type[PackedTensor] | type[PaddedTensor], invalid: str
) -> None:
    """Reject ambiguous batches before constructing a record."""
    items: list[torch.Tensor]
    match invalid:
        case "empty":
            items = []
        case "rank":
            items = [torch.ones(2)]
        case "width":
            items = [torch.ones(2, 3), torch.ones(2, 4)]
        case _:
            items = [torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64)]
    with pytest.raises(ValueError):
        tensor_type.from_list(items)


@pytest.mark.parametrize("invalid", ["count", "length", "width", "rank"])
def test_tensor_rejects_invalid_positions(invalid: str) -> None:
    """Explicit coordinate lists must align with every value tensor."""
    items = [torch.ones(2, 3), torch.ones(1, 3)]
    match invalid:
        case "count":
            positions = [torch.ones(2, 1)]
        case "length":
            positions = [torch.ones(2, 1), torch.ones(2, 1)]
        case "width":
            positions = [torch.ones(2, 1), torch.ones(1, 2)]
        case _:
            positions = [torch.ones(2), torch.ones(1)]
    with pytest.raises(ValueError):
        PackedTensor.from_list(items, abs_positions=positions)


def test_tensor_alias_deprecation() -> None:
    """Both import locations warn while returning the exact canonical classes."""
    import stackformers
    import stackformers.sequence as sequence

    for module in (stackformers, sequence):
        for old_name, new_type in (("PackedInput", PackedTensor), ("PaddedInput", PaddedTensor)):
            with pytest.warns(DeprecationWarning, match=new_type.__name__):
                assert getattr(module, old_name) is new_type
    assert "__deprecated__" in vars(make_packed_input)
    assert "__deprecated__" in vars(PackedSequence)


def test_tensor_all_valid_views(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    """All-valid conversions avoid allocating masks and share contiguous value storage."""
    device, dtype = device_dtype
    x = torch.randn(2, 3, 8, device=device, dtype=dtype)
    positions = torch.zeros(2, 3, 1, device=device)
    tensor = PaddedTensor(x, AllValid(), positions)
    assert tensor.to_packed().x.data_ptr() == x.data_ptr()
    assert tensor.to_list()[0].data_ptr() == x.data_ptr()


class _PackedToPaddedWrapper(nn.Module):
    """Expose the packed conversion method through a traceable module boundary."""

    @override
    def forward(self, input: PackedTensor) -> PaddedTensor:
        """Return the right-padded representation without Python length iteration."""
        return input.to_padded()


def test_tensor_method_export(packed_input_ragged: PackedInput) -> None:
    """Canonical records and the padded conversion method survive torch.export."""
    module = _PackedToPaddedWrapper()
    exported = torch.export.export(module, (packed_input_ragged,))
    actual = exported.module()(packed_input_ragged)
    expected = module(packed_input_ragged)
    torch.testing.assert_close(actual, expected)


def test_tensor_method_compile(packed_input_ragged: PackedInput) -> None:
    """The conversion traces as a full graph with dynamic-output capture enabled."""
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(_PackedToPaddedWrapper(), backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(packed_input_ragged), packed_input_ragged.to_padded())
