"""Tests for decoder stacks and normalization topologies."""

from __future__ import annotations

import pytest
import torch

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.decoder import (
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.norm.config import NormPlacement, RMSNormConfig
from stackformers.positional.config import NoPosEncodingConfig, RoPE1DConfig
from stackformers.presets.decoder import TransformerDecoder, TransformerDecoderConfig
from stackformers.sequence import PaddedInput, make_padded_input
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run
from tests.norm_topology_helpers import (
    AffineNorm,
    ScaleCrossAttention,
    ScaleFeedForward,
    ScaleSelfAttention,
    expected_cross_attention,
    expected_norm,
    make_topology_input,
)

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


def _decoder_layer(norm_placement: NormPlacement) -> DecoderLayerBase:
    """Build a deterministic decoder layer for an exact topology check."""
    self_attn = ScaleSelfAttention(2.0)
    cross_attn = ScaleCrossAttention(3.0)
    ff = ScaleFeedForward(4.0)
    match norm_placement:
        case "pre":
            return DecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "post":
            return PostNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "sandwich":
            return SandwichNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "reordered":
            return ReorderedNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )


def _expected_decoder_output(
    norm_placement: NormPlacement,
    x: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the selected decoder equation independently of its implementation."""
    match norm_placement:
        case "pre":
            after_self = x + 2.0 * expected_norm(x)
            after_cross = after_self + expected_cross_attention(expected_norm(after_self), context)
            return after_cross + 4.0 * expected_norm(after_cross)
        case "post":
            after_self = expected_norm(x + 2.0 * x)
            after_cross = expected_norm(after_self + expected_cross_attention(after_self, context))
            return expected_norm(after_cross + 4.0 * after_cross)
        case "sandwich":
            after_self = x + expected_norm(2.0 * expected_norm(x))
            after_cross = after_self + expected_norm(
                expected_cross_attention(expected_norm(after_self), context)
            )
            return after_cross + expected_norm(4.0 * expected_norm(after_cross))
        case "reordered":
            after_self = x + expected_norm(2.0 * x)
            after_cross = after_self + expected_norm(expected_cross_attention(after_self, context))
            return after_cross + expected_norm(4.0 * after_cross)


@pytest.fixture
def config() -> TransformerDecoderConfig:
    return TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )


@pytest.fixture
def decoder(
    config: TransformerDecoderConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> TransformerDecoder:
    device, dtype = device_dtype
    return TransformerDecoder(config).to(device=device, dtype=dtype)


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


def test_decoder_output_shape(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    x_inp, ctx_inp = x_context_inp
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_ctx_padding(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x_inp, _ = x_context_inp
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[1, 8:] = False
    ctx_inp = make_padded_input(context, mask)
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_tgt_padding(
    decoder: TransformerDecoder,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 6:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_causal_self_attn_shape() -> None:
    """Decoder with causal=True in self_attn config runs without error."""
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=1,
    )
    model = TransformerDecoder(cfg)
    x = torch.randn(B, N, D)
    context = torch.randn(B, S, D)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_rope(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=RoPE1DConfig(dim_head=DH),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_gradients(device: torch.device) -> None:
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    context = torch.randn(B, S, D, device=device, requires_grad=True)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    model(x_inp, ctx_inp).sum().backward()
    assert x.grad is not None
    assert context.grad is not None


def test_decoder_config_accessor(config: TransformerDecoderConfig) -> None:
    assert TransformerDecoder(config).config is config


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_norm_placement_equation(norm_placement: NormPlacement) -> None:
    """Each decoder class implements its exact three-branch normalization equation."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _decoder_layer(norm_placement)

    expected = _expected_decoder_output(norm_placement, x_input.x, ctx_input.x)

    assert torch.equal(layer(x_input, ctx_input).x, expected)


def test_sandwich_decoder_uses_six_independent_norms() -> None:
    """No affine scale is tied across sandwich branch boundaries."""
    layer = _decoder_layer("sandwich")

    assert isinstance(layer, SandwichNormDecoderLayer)
    norms = [
        layer.norm_self_pre,
        layer.norm_self_post,
        layer.norm_cross_pre,
        layer.norm_cross_post,
        layer.norm_ff_pre,
        layer.norm_ff_post,
    ]
    assert len({id(norm) for norm in norms}) == 6


def test_decoder_layer_retains_legacy_state_dict_keys() -> None:
    """The pre-norm decoder constructor and checkpoint names remain unchanged."""
    layer = _decoder_layer("pre")

    assert list(layer.state_dict()) == [
        "self_attn.scale",
        "cross_attn.query_scale",
        "cross_attn.context_scale",
        "ff.scale",
        "norm_self.scale",
        "norm_self.offset",
        "norm_cross.scale",
        "norm_cross.offset",
        "norm_ff.scale",
        "norm_ff.offset",
    ]


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_norm_placement_gradients(norm_placement: NormPlacement) -> None:
    """Every decoder topology propagates gradients through target and context inputs."""
    x_input = make_topology_input(requires_grad=True)
    ctx_input = make_topology_input(12, requires_grad=True)
    layer = _decoder_layer(norm_placement)

    layer(x_input, ctx_input).x.sum().backward()

    assert x_input.x.grad is not None
    assert ctx_input.x.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_decoder_norm_placement_is_export_compatible(
    norm_placement: NormPlacement,
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Every decoder topology supports static and dynamic export."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _decoder_layer(norm_placement)
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
