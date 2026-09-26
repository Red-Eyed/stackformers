"""ONNX Runtime regression coverage for broadcast padding masks in attention."""

from typing import NamedTuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from stackformers.attention.ops import padded_sdpa, padding_mask
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run


class AttentionInputs(NamedTuple):
    """Keep query and context lengths independent at the export boundary."""

    q: Tensor
    k: Tensor
    v: Tensor
    mask: Tensor


class PaddedAttention(nn.Module):
    """Expose the shared padding-only attention path without projection layers."""

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """Attend to valid context tokens with multiple query heads."""
        return padded_sdpa(q, k, v, mask, causal=False, window_size=None, bias=None)


def _attention_inputs(batch: int, queries: int, keys: int) -> AttentionInputs:
    """Build reproducible attention inputs with different padding in each batch item."""
    generator = torch.Generator().manual_seed(7)
    mask = torch.ones(batch, keys, dtype=torch.bool)
    mask[0, -2:] = False
    mask[1:, -1:] = False
    return AttentionInputs(
        torch.randn(batch, 2, queries, 8, generator=generator),
        torch.randn(batch, 2, keys, 8, generator=generator),
        torch.randn(batch, 2, keys, 8, generator=generator),
        mask,
    )


@pytest.fixture(params=[False, True], ids=["partial-padding", "empty-context"])
def empty_context(request: pytest.FixtureRequest) -> bool:
    """Include an empty document alongside nonempty documents in each export mode."""
    assert isinstance(request.param, bool)
    return request.param


@pytest.fixture
def example_inputs(empty_context: bool) -> AttentionInputs:
    """Use non-singleton dimensions so dynamic export can retain each axis."""
    inputs = _attention_inputs(2, 3, 5)
    if empty_context:
        inputs.mask[0] = False
    return inputs


@pytest.fixture(params=[1, 4], ids=["single-query", "multiple-queries"])
def runtime_inputs(request: pytest.FixtureRequest, empty_context: bool) -> AttentionInputs:
    """Change batch, query, and context lengths independently of the export example."""
    inputs = _attention_inputs(3, request.param, 6)
    if empty_context:
        inputs.mask[0] = False
    return inputs


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
def test_padding_mask_export(
    example_inputs: AttentionInputs,
    runtime_inputs: AttentionInputs,
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Explicit mask expansion preserves broadcast SDPA results in ONNX Runtime."""
    batch = torch.export.Dim("batch", min=1, max=4)
    queries = torch.export.Dim("queries", min=1, max=8)
    keys = torch.export.Dim("keys", min=1, max=8)
    (actual,) = export_and_run(
        PaddedAttention(),
        tuple(example_inputs),
        shape_mode,
        opset_version,
        dynamic_shapes=(
            {0: batch, 2: queries},
            {0: batch, 2: keys},
            {0: batch, 2: keys},
            {0: batch, 1: keys},
        ),
        runtime_args=tuple(runtime_inputs),
    )
    inputs = runtime_inputs if shape_mode.is_dynamic else example_inputs
    # Compare against the original broadcast-mask semantics, independently of
    # padded_sdpa, so export/eager agreement alone cannot hide a masking change.
    expected = F.scaled_dot_product_attention(
        inputs.q, inputs.k, inputs.v, attn_mask=padding_mask(inputs.mask, inputs.q.dtype)
    )
    torch.testing.assert_close(actual, expected)
