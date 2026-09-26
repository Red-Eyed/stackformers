"""Tests for cross-attender stacks and normalization topologies."""

from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import CrossAttentionConfig
from stackformers.cross_attender import (
    CrossAttenderLayer,
    CrossAttenderLayerBase,
    PostNormCrossAttenderLayer,
    ReorderedNormCrossAttenderLayer,
    SandwichNormCrossAttenderLayer,
)
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.norm.config import NormPlacement, RMSNormConfig
from stackformers.presets.cross_attender import CrossAttender, CrossAttenderConfig
from stackformers.sequence import PaddedInput, make_padded_input
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run
from tests.norm_topology_helpers import (
    AffineNorm,
    ScaleCrossAttention,
    ScaleFeedForward,
    expected_cross_attention,
    expected_norm,
    make_topology_input,
)

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


def _cross_attender_layer(norm_placement: NormPlacement) -> CrossAttenderLayerBase:
    """Build a deterministic cross-attender layer for an exact topology check."""
    cross_attn = ScaleCrossAttention(3.0)
    ff = ScaleFeedForward(4.0)
    match norm_placement:
        case "pre":
            return CrossAttenderLayer(
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "post":
            return PostNormCrossAttenderLayer(
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "sandwich":
            return SandwichNormCrossAttenderLayer(
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "reordered":
            return ReorderedNormCrossAttenderLayer(
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )


def _expected_cross_attender_output(
    norm_placement: NormPlacement,
    x: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the selected cross-attender equation independently of its implementation."""
    match norm_placement:
        case "pre":
            after_cross = x + expected_cross_attention(expected_norm(x), context)
            return after_cross + 4.0 * expected_norm(after_cross)
        case "post":
            after_cross = expected_norm(x + expected_cross_attention(x, context))
            return expected_norm(after_cross + 4.0 * after_cross)
        case "sandwich":
            after_cross = x + expected_norm(expected_cross_attention(expected_norm(x), context))
            return after_cross + expected_norm(4.0 * expected_norm(after_cross))
        case "reordered":
            after_cross = x + expected_norm(expected_cross_attention(x, context))
            return after_cross + expected_norm(4.0 * after_cross)


@pytest.fixture
def config() -> CrossAttenderConfig:
    return CrossAttenderConfig(
        attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=2,
    )


@pytest.fixture
def cross_attender(
    config: CrossAttenderConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> CrossAttender:
    device, dtype = device_dtype
    return CrossAttender(config).to(device=device, dtype=dtype)


@pytest.fixture
def x_context_inp(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[PaddedInput, PaddedInput]:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    return x_inp, ctx_inp


def test_cross_attender_output_shape(
    cross_attender: CrossAttender,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    x_inp, ctx_inp = x_context_inp
    out = cross_attender(x_inp, ctx_inp)
    assert out.shape == (B, N, D)


def test_cross_attender_with_ctx_padding(
    cross_attender: CrossAttender,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[1, 8:] = False
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, mask)
    assert cross_attender(x_inp, ctx_inp).shape == (B, N, D)


def test_cross_attender_with_x_padding(
    cross_attender: CrossAttender,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 5:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert cross_attender(x_inp, ctx_inp).shape == (B, N, D)


def test_cross_attender_gradients(device: torch.device) -> None:
    cfg = CrossAttenderConfig(
        attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=2,
    )
    model = CrossAttender(cfg).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    context = torch.randn(B, S, D, device=device, requires_grad=True)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    model(x_inp, ctx_inp).sum().backward()
    assert x.grad is not None
    assert context.grad is not None


def test_cross_attender_config_accessor(config: CrossAttenderConfig) -> None:
    model = CrossAttender(config)
    assert model.config is config


def test_cross_attender_gqa(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    cfg = CrossAttenderConfig(
        attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH, kv_heads=2),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        num_layers=1,
    )
    model = CrossAttender(cfg).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_cross_attender_norm_placement_equation(norm_placement: NormPlacement) -> None:
    """Each cross-attender class implements its exact two-branch normalization equation."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _cross_attender_layer(norm_placement)

    expected = _expected_cross_attender_output(norm_placement, x_input.x, ctx_input.x)

    assert torch.equal(layer(x_input, ctx_input).x, expected)


def test_sandwich_cross_attender_uses_four_independent_norms() -> None:
    """No affine scale is tied across sandwich branch boundaries."""
    layer = _cross_attender_layer("sandwich")

    assert isinstance(layer, SandwichNormCrossAttenderLayer)
    norms = [
        layer.norm_cross_pre,
        layer.norm_cross_post,
        layer.norm_ff_pre,
        layer.norm_ff_post,
    ]
    assert len({id(norm) for norm in norms}) == 4


def test_cross_attender_layer_retains_legacy_state_dict_keys() -> None:
    """The pre-norm cross-attender constructor and checkpoint names remain unchanged."""
    layer = _cross_attender_layer("pre")

    assert list(layer.state_dict()) == [
        "cross_attn.query_scale",
        "cross_attn.context_scale",
        "ff.scale",
        "norm_cross.scale",
        "norm_cross.offset",
        "norm_ff.scale",
        "norm_ff.offset",
    ]


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_cross_attender_norm_placement_gradients(norm_placement: NormPlacement) -> None:
    """Every topology propagates gradients through query and context inputs."""
    x_input = make_topology_input(requires_grad=True)
    ctx_input = make_topology_input(12, requires_grad=True)
    layer = _cross_attender_layer(norm_placement)

    layer(x_input, ctx_input).x.sum().backward()

    assert x_input.x.grad is not None
    assert ctx_input.x.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_cross_attender_norm_placement_is_export_compatible(
    norm_placement: NormPlacement,
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Every cross-attender topology supports static and dynamic export."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _cross_attender_layer(norm_placement)
    resized_x_input = make_padded_input(
        torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )
    resized_ctx_input = make_padded_input(
        torch.arange(12, 28, dtype=torch.float32).reshape(1, 4, 4),
        torch.ones(1, 4, dtype=torch.bool),
    )
    query_tokens = torch.export.Dim("query_tokens", min=1, max=4)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=5)
    shapes = torch.export.ShapesCollection()
    for tensor in x_input:
        shapes[tensor] = {1: query_tokens}
    for tensor in ctx_input:
        shapes[tensor] = {1: context_tokens}

    export_and_run(
        layer,
        (x_input, ctx_input),
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(layer, (x_input, ctx_input)),
        runtime_args=(resized_x_input, resized_ctx_input),
    )
