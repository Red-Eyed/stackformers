"""Tests for transformer-layer execution and normalization placement."""

from __future__ import annotations

import copy
import io

import pytest
import torch

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import (
    NormPlacement,
    PostNormTransformerLayer,
    ReorderedNormTransformerLayer,
    SandwichNormTransformerLayer,
    TransformerLayer,
    TransformerLayerBase,
)
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PaddedInput, make_padded_input
from tests.norm_topology_helpers import (
    AffineNorm,
    ScaleFeedForward,
    ScaleSelfAttention,
    make_topology_input,
)

B, N, D, H, DH = 2, 16, 64, 4, 16


def _order_test_layer(norm_placement: NormPlacement = "pre") -> TransformerLayerBase:
    """Build a deterministic layer whose four placement equations differ exactly."""
    self_attn = ScaleSelfAttention(2.0)
    ff = ScaleFeedForward(3.0)
    match norm_placement:
        case "pre":
            return TransformerLayer(
                self_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "post":
            return PostNormTransformerLayer(
                self_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "sandwich":
            return SandwichNormTransformerLayer(
                self_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "reordered":
            return ReorderedNormTransformerLayer(
                self_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )


def _legacy_order_test_layer() -> TransformerLayer:
    """Build the deterministic layer through the original four-argument constructor."""
    return TransformerLayer(
        ScaleSelfAttention(2.0),
        ScaleFeedForward(3.0),
        AffineNorm(5.0, 1.0),
        AffineNorm(5.0, 1.0),
    )


def _order_test_input(*, requires_grad: bool = False) -> PaddedInput:
    """Build a small deterministic padded input for placement tests."""
    return make_topology_input(requires_grad=requires_grad)


@pytest.fixture
def layer(device_dtype: tuple[torch.device, torch.dtype]) -> TransformerLayer:
    device, dtype = device_dtype
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    norm_cfg = RMSNormConfig(dim=D)
    return TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding()),
        ff=SwiGLU(ff_cfg),
        norm_attn=build_norm(norm_cfg),
        norm_ff=build_norm(norm_cfg),
    ).to(device=device, dtype=dtype)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_transformer_layer_output_shape(
    layer: TransformerLayer,
    x_pad: PaddedInput,
) -> None:
    out = layer(x_pad)
    assert out.x.shape == (B, N, D)


def test_transformer_layer_residual_connection(
    layer: TransformerLayer,
    x_pad: PaddedInput,
) -> None:
    out = layer(x_pad)
    assert not torch.allclose(out.x, x_pad.x)


def test_transformer_layer_gradients(device: torch.device) -> None:
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    norm_cfg = RMSNormConfig(dim=D)
    layer = TransformerLayer(
        self_attn=SelfAttention(attn_cfg, NoPosEncoding()),
        ff=SwiGLU(ff_cfg),
        norm_attn=build_norm(norm_cfg),
        norm_ff=build_norm(norm_cfg),
    ).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    inp = make_padded_input(x, mask)
    layer(inp).x.sum().backward()
    assert x.grad is not None


def test_default_norm_placement_preserves_pre_norm_equation() -> None:
    """Omitting the new argument remains exactly equivalent to the legacy equation."""
    input = _order_test_input()
    layer = _legacy_order_test_layer()

    normed = 5.0 * input.x + 1.0
    after_attention = input.x + 2.0 * normed
    expected = after_attention + 3.0 * (5.0 * after_attention + 1.0)

    assert isinstance(layer, TransformerLayer)
    assert torch.equal(layer(input).x, expected)


def test_reordered_norm_normalizes_branch_outputs() -> None:
    """Reordered norm applies after each branch and before its residual addition."""
    input = _order_test_input()
    layer = _order_test_layer("reordered")

    after_attention = input.x + (5.0 * (2.0 * input.x) + 1.0)
    expected = after_attention + (5.0 * (3.0 * after_attention) + 1.0)

    assert torch.equal(layer(input).x, expected)


def test_post_norm_normalizes_residual_sums() -> None:
    """Post-norm applies after adding each unnormalized branch to its residual."""
    input = _order_test_input()
    layer = _order_test_layer("post")

    after_attention = 5.0 * (input.x + 2.0 * input.x) + 1.0
    expected = 5.0 * (after_attention + 3.0 * after_attention) + 1.0

    assert torch.equal(layer(input).x, expected)


def test_sandwich_norm_normalizes_both_sides_with_distinct_modules() -> None:
    """Sandwich norm uses independent pre- and post-branch affine parameters."""
    input = _order_test_input()
    layer = _order_test_layer("sandwich")

    after_attention = input.x + (5.0 * (2.0 * (5.0 * input.x + 1.0)) + 1.0)
    expected = after_attention + (5.0 * (3.0 * (5.0 * after_attention + 1.0)) + 1.0)

    assert layer.norm_attn_post is not layer.norm_attn_pre
    assert layer.norm_ff_post is not layer.norm_ff_pre
    assert torch.equal(layer(input).x, expected)


def test_explicit_pre_norm_matches_default_outputs_and_gradients() -> None:
    """The explicit and omitted pre-norm settings are numerically identical."""
    default_layer = _legacy_order_test_layer()
    explicit_layer = _order_test_layer("pre")
    explicit_layer.load_state_dict(copy.deepcopy(default_layer.state_dict()))
    default_input = _order_test_input(requires_grad=True)
    explicit_input = default_input._replace(x=default_input.x.detach().clone().requires_grad_())

    default_output = default_layer(default_input).x
    explicit_output = explicit_layer(explicit_input).x
    default_output.sum().backward()
    explicit_output.sum().backward()

    assert torch.equal(default_output, explicit_output)
    assert default_input.x.grad is not None
    assert explicit_input.x.grad is not None
    assert torch.equal(default_input.x.grad, explicit_input.x.grad)
    for (_, default_parameter), (_, explicit_parameter) in zip(
        default_layer.named_parameters(), explicit_layer.named_parameters(), strict=True
    ):
        assert default_parameter.grad is not None
        assert explicit_parameter.grad is not None
        assert torch.equal(default_parameter.grad, explicit_parameter.grad)


def test_pre_norm_state_dict_retains_legacy_keys() -> None:
    """Moving shared branches into the base class does not rename checkpoint entries."""
    layer = _legacy_order_test_layer()

    assert list(layer.state_dict()) == [
        "self_attn.scale",
        "ff.scale",
        "norm_attn.scale",
        "norm_attn.offset",
        "norm_ff.scale",
        "norm_ff.offset",
    ]


@pytest.mark.parametrize("norm_placement", ["post", "reordered"])
def test_non_sandwich_placement_does_not_change_state_dict_contract(
    norm_placement: NormPlacement,
) -> None:
    """Single-norm layouts retain the legacy parameter names and shapes."""
    pre_layer = _legacy_order_test_layer()
    alternative_layer = _order_test_layer(norm_placement)

    assert pre_layer.state_dict().keys() == alternative_layer.state_dict().keys()
    alternative_layer.load_state_dict(pre_layer.state_dict(), strict=True)
    pre_layer.load_state_dict(alternative_layer.state_dict(), strict=True)


def test_legacy_whole_module_without_placement_defaults_to_pre_norm() -> None:
    """The unchanged pre-norm class retains whole-module checkpoint compatibility."""
    legacy_layer = _legacy_order_test_layer()
    checkpoint = io.BytesIO()
    torch.save(legacy_layer, checkpoint)
    checkpoint.seek(0)

    restored = torch.load(checkpoint, weights_only=False)

    assert isinstance(restored, TransformerLayer)
    assert torch.equal(
        restored(_order_test_input()).x,
        _legacy_order_test_layer()(_order_test_input()).x,
    )


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_norm_placement_is_torch_export_compatible(
    norm_placement: NormPlacement,
) -> None:
    """Every placement remains traceable through the library's promised export path."""
    layer: TransformerLayerBase = _order_test_layer(norm_placement)
    input = _order_test_input()

    exported = torch.export.export(layer, (input,))

    assert torch.equal(exported.module()(input).x, layer(input).x)
