"""Packed QKV parity across head sharing, layouts, visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.config import RoPE1DConfig
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.sequence import PaddedInput, SequenceInput, make_packed_input, make_padded_input
from tests.conftest import atol
from tests.export_utils import ExportShapeMode, export_and_run

if TYPE_CHECKING:
    from stackformers.attention.masking import SelfAttentionMask


@pytest.fixture(params=[1, 2, 4])
def source(
    request: pytest.FixtureRequest, device_dtype: tuple[torch.device, torch.dtype]
) -> SelfAttention:
    """Exercise MQA, GQA, and MHA with rotary positions and QK normalization."""
    device, dtype = device_dtype
    config = SelfAttentionConfig(
        dim=32, heads=4, dim_head=8, kv_heads=request.param, causal=True, qk_norm=True
    )
    return (
        SelfAttention(config, RotaryEmbedding1D(RoPE1DConfig(dim_head=8)))
        .to(device=device, dtype=dtype)
        .eval()
    )


@pytest.fixture(params=[False, True], ids=["padded", "packed"])
def sequence(request: pytest.FixtureRequest, source: SelfAttention) -> SequenceInput:
    """Represent two unequal documents in either supported layout."""
    weight = source.to_q.weight
    if request.param:
        x = torch.randn(8, 32, device=weight.device, dtype=weight.dtype)
        cu = torch.tensor([0, 5, 8], device=weight.device, dtype=torch.int32)
        return make_packed_input(x, cu, max_seqlen=5)
    x = torch.randn(2, 5, 32, device=weight.device, dtype=weight.dtype)
    mask = (
        torch.arange(5, device=weight.device)[None]
        < torch.tensor([5, 3], device=weight.device)[:, None]
    )
    return make_padded_input(x, mask)


@pytest.mark.parametrize("visibility", ["default", "causal", "bidirectional", "explicit"])
def test_forward_parity(source: SelfAttention, sequence: SequenceInput, visibility: str) -> None:
    """Packing preserves position, normalization, masks, and output order."""
    packed = source.to_packed()
    mask: SelfAttentionMask
    match visibility:
        case "default" | "causal" | "bidirectional":
            mask = visibility
        case _:
            mask = torch.eye(5, dtype=torch.bool, device=sequence.x.device)
    expected = source(sequence, self_attn_mask=mask)
    actual = packed(sequence, self_attn_mask=mask)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=atol(actual.dtype))
    for name, value in source.state_dict().items():
        torch.testing.assert_close(packed.unpack().state_dict()[name], value, rtol=0, atol=0)


def test_gradients(device: torch.device) -> None:
    """Unequal Q/KV row groups receive the original projection gradients."""
    source = SelfAttention(
        SelfAttentionConfig(dim=32, heads=4, dim_head=8, kv_heads=2, qk_norm=True),
        RotaryEmbedding1D(RoPE1DConfig(dim_head=8)),
    ).to(device)
    packed = source.to_packed()
    x = torch.randn(2, 5, 32, device=device, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    mask = torch.ones(2, 5, device=device, dtype=torch.bool)
    source(make_padded_input(x, mask)).square().sum().backward()
    packed(make_padded_input(other, mask)).square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad)
    assert packed.qkv.weight.grad is not None
    for actual, layer in zip(
        packed.qkv.weight.grad.split((32, 16, 16)),
        (source.to_q, source.to_k, source.to_v),
        strict=True,
    ):
        torch.testing.assert_close(actual, layer.weight.grad)
    for name in ("to_out.weight", "q_norm.weight", "k_norm.weight"):
        torch.testing.assert_close(packed.get_parameter(name).grad, source.get_parameter(name).grad)


def test_dynamic_export(source: SelfAttention) -> None:
    """One exported graph handles singleton and longer sequence axes."""
    packed = source.to_packed()
    weight = source.to_q.weight
    example = make_padded_input(
        torch.randn(2, 3, 32, device=weight.device, dtype=weight.dtype),
        torch.ones(2, 3, device=weight.device, dtype=torch.bool),
    )
    shapes = torch.export.ShapesCollection()
    length = torch.export.Dim("tokens", min=1, max=32)
    for tensor in example:
        shapes[tensor] = {1: length}
    program = torch.export.export(
        packed, (example,), dynamic_shapes=shapes.dynamic_shapes(packed, (example,))
    ).module()
    for tokens in (1, 2, 7):
        sequence = make_padded_input(
            torch.randn(2, tokens, 32, device=weight.device, dtype=weight.dtype),
            torch.ones(2, tokens, device=weight.device, dtype=torch.bool),
        )
        torch.testing.assert_close(program(sequence), packed(sequence))


def test_window_and_bias(source: SelfAttention, sequence: SequenceInput) -> None:
    """Learned bias and local visibility survive conversion in either sequence layout."""
    from stackformers.attention.config import DistanceBiasConfig
    from stackformers.attention.distance_bias import RelativeDistanceBias

    source.config = source.config.model_copy(update={"window_size": 3})
    source.attn_bias = RelativeDistanceBias(DistanceBiasConfig(heads=4, r_max=10, num_rbf=4)).to(
        device=sequence.x.device, dtype=sequence.x.dtype
    )
    packed = source.to_packed()
    expected = source(sequence)
    actual = packed(sequence)
    torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=atol(actual.dtype))


@pytest.fixture(params=[1, 2, 4], ids=["mqa", "gqa", "mha"])
def onnx_source(request: pytest.FixtureRequest) -> SelfAttention:
    """Export CPU float32 projections across all supported head-sharing geometries."""
    return SelfAttention(
        SelfAttentionConfig(dim=32, heads=4, dim_head=8, kv_heads=request.param, qk_norm=True),
        RotaryEmbedding1D(RoPE1DConfig(dim_head=8)),
    ).eval()


@pytest.fixture
def onnx_query() -> PaddedInput:
    """Trace a non-singleton query with a partially padded document."""
    return make_padded_input(
        torch.randn(2, 3, 32), torch.tensor([[True, True, True], [True, True, False]])
    )


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", range(20, 26))
@pytest.mark.parametrize("pack_weights", [False, True], ids=["ordinary-weights", "packed-weights"])
def test_onnx_export(
    onnx_source: SelfAttention,
    onnx_query: PaddedInput,
    shape_mode: ExportShapeMode,
    opset_version: int,
    pack_weights: bool,
) -> None:
    """Both projection layouts preserve rotary attention through mandatory ONNX execution."""
    model = onnx_source.to_packed() if pack_weights else onnx_source
    resized_query = make_padded_input(torch.randn(2, 1, 32), torch.ones(2, 1, dtype=torch.bool))
    shapes = torch.export.ShapesCollection()
    queries = torch.export.Dim("queries", min=1, max=8)
    for tensor in onnx_query:
        shapes[tensor] = {1: queries}
    example_args = (onnx_query,)
    runtime_args = (resized_query,)
    export_and_run(
        model,
        example_args,
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(model, example_args),
        runtime_args=runtime_args,
    )
