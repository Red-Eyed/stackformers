"""Tests for cross-attender presets and normalization-topology selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from pydantic import ValidationError

from stackformers.cross_attender import (
    CrossAttenderLayer,
    CrossAttenderLayerBase,
    PostNormCrossAttenderLayer,
    ReorderedNormCrossAttenderLayer,
    SandwichNormCrossAttenderLayer,
)
from stackformers.presets.cross_attender import (
    CrossAttender,
    CrossAttenderConfig,
    plain_cross_attender_config,
)
from stackformers.sequence import PaddedInput, make_padded_input
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run

if TYPE_CHECKING:
    from stackformers.norm.config import NormPlacement

B, Nq, S, D, H = 2, 8, 12, 64, 4  # Nq=query len, S=context len


@pytest.fixture
def query_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, Nq, D, device=device, dtype=dtype)
    mask = torch.ones(B, Nq, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def context_input(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


def test_plain_cross_attender_output_shape(
    device_dtype: tuple[torch.device, torch.dtype],
    query_input: PaddedInput,
    context_input: PaddedInput,
) -> None:
    device, dtype = device_dtype
    cfg = plain_cross_attender_config(D, H, num_layers=2)
    out = CrossAttender(cfg).to(device=device, dtype=dtype)(query_input, context_input)
    assert out.shape == (B, Nq, D)


def test_legacy_cross_attender_config_defaults_to_pre_norm() -> None:
    """A serialized config without placement keeps the original cross-attender topology."""
    legacy_payload = plain_cross_attender_config(D, H, num_layers=2).model_dump()
    legacy_payload.pop("norm_placement")

    restored = CrossAttenderConfig.model_validate(legacy_payload)

    assert restored.norm_placement == "pre"


@pytest.mark.parametrize(
    ("norm_placement", "layer_type"),
    [
        ("pre", CrossAttenderLayer),
        ("post", PostNormCrossAttenderLayer),
        ("sandwich", SandwichNormCrossAttenderLayer),
        ("reordered", ReorderedNormCrossAttenderLayer),
    ],
)
def test_cross_attender_builds_selected_norm_topology(
    norm_placement: NormPlacement,
    layer_type: type[CrossAttenderLayerBase],
) -> None:
    """Every supported config value selects its focused cross-attender implementation."""
    config = plain_cross_attender_config(
        D,
        H,
        num_layers=2,
        norm_placement=norm_placement,
    )

    cross_attender = CrossAttender(config)

    assert all(isinstance(layer, layer_type) for layer in cross_attender._stack.layers)


def test_cross_attender_config_rejects_unknown_norm_placement() -> None:
    """The shared Literal rejects unsupported cross-attender topology names."""
    payload = plain_cross_attender_config(D, H, num_layers=2).model_dump()
    payload["norm_placement"] = "future"

    with pytest.raises(ValidationError):
        CrossAttenderConfig.model_validate(payload)


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_plain_cross_attender_is_export_compatible(
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """The preset exports static and independently dynamic query/context lengths."""
    model = CrossAttender(plain_cross_attender_config(D, heads=1, num_layers=1))
    query = make_padded_input(
        torch.randn(2, Nq, D),
        torch.ones(2, Nq, dtype=torch.bool),
    )
    context = make_padded_input(
        torch.randn(2, S, D),
        torch.ones(2, S, dtype=torch.bool),
    )
    resized_query = make_padded_input(
        torch.randn(1, Nq // 2, D),
        torch.ones(1, Nq // 2, dtype=torch.bool),
    )
    resized_context = make_padded_input(
        torch.randn(1, S // 2, D),
        torch.ones(1, S // 2, dtype=torch.bool),
    )
    batch = torch.export.Dim("batch", min=1, max=B)
    query_tokens = torch.export.Dim("query_tokens", min=1, max=Nq)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=S)
    shapes = torch.export.ShapesCollection()
    for tensor in query:
        shapes[tensor] = {0: batch, 1: query_tokens}
    for tensor in context:
        shapes[tensor] = {0: batch, 1: context_tokens}

    export_and_run(
        model,
        (query, context),
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(model, (query, context)),
        runtime_args=(resized_query, resized_context),
    )
