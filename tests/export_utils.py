"""Shared execution and parity checks for PyTorch and ONNX exports."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

import pytest
import torch
import torch.nn as nn
from onnx import defs
from onnxscript import ir
from torch.utils import _pytree

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

# Export shape trees mirror arbitrary module inputs; torch.export validates their leaves.
DynamicShapes: TypeAlias = dict[str, object] | tuple[object, ...] | list[object] | None
# The dynamo exporter supports opsets beyond the legacy exporter's ONNX_MAX_OPSET.
# Cap coverage at the highest verified target; newer requests may silently retain
# an older opset, which _assert_onnx_opset_version deliberately rejects.
ONNX_OPSET_VERSIONS = tuple(range(15, min(defs.onnx_opset_version(), 25) + 1))
REQUIRED_ONNX_OPSET_VERSIONS = tuple(
    version for version in ONNX_OPSET_VERSIONS if 18 <= version <= 25
)
OPTIONAL_ONNX_OPSET_VERSIONS = tuple(
    version for version in ONNX_OPSET_VERSIONS if version not in REQUIRED_ONNX_OPSET_VERSIONS
)


def _optional_onnx_opset_reason(opset_version: int) -> str:
    """Describe why an opset remains compatibility coverage rather than a requirement."""
    if opset_version < 18:
        return "PyTorch's ONNX C API cannot down-convert Transformer reductions below 18"
    return "This installed opset has not been promoted to required compatibility"


def _onnx_opset_case(opset_version: int) -> ParameterSet:
    """Build one required or optional opset compatibility case."""
    if opset_version in REQUIRED_ONNX_OPSET_VERSIONS:
        return pytest.param(opset_version, id=f"required-opset-{opset_version}")
    return pytest.param(
        opset_version,
        marks=pytest.mark.xfail(
            reason=_optional_onnx_opset_reason(opset_version),
            strict=False,
        ),
        id=f"optional-opset-{opset_version}",
    )


ONNX_OPSET_CASES = tuple(_onnx_opset_case(version) for version in ONNX_OPSET_VERSIONS)


class ExportShapeMode(StrEnum):
    """Select whether exported input dimensions are concrete or symbolic."""

    STATIC = "static"
    DYNAMIC = "dynamic"

    @property
    def is_dynamic(self) -> bool:
        """Return whether the exporter should retain the supplied dynamic dimensions."""
        return self is self.DYNAMIC


def _assert_onnx_shape_mode(program: torch.onnx.ONNXProgram, mode: ExportShapeMode) -> None:
    """Verify that ONNX graph inputs expose concrete or symbolic dimensions as requested."""
    dimensions = [
        dimension
        for value in program.model.graph.inputs
        if value.shape is not None
        for dimension in value.shape
    ]
    if mode.is_dynamic:
        assert any(isinstance(dimension, ir.SymbolicDim) for dimension in dimensions)
    else:
        assert all(isinstance(dimension, int) for dimension in dimensions)


def _assert_onnx_opset_version(
    program: torch.onnx.ONNXProgram,
    opset_version: int,
) -> None:
    """Verify that down-conversion produced the requested standard ONNX opset."""
    assert program.model.opset_imports[""] == opset_version


def _flatten_tensor_output(output: object) -> tuple[torch.Tensor, ...]:
    """Flatten an exported output tree and require tensor-only leaves."""
    leaves, _ = _pytree.tree_flatten(output)
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    return tuple(leaves)


def export_and_run(
    model: nn.Module,
    example_args: tuple[object, ...],
    mode: ExportShapeMode,
    opset_version: int,
    *,
    dynamic_shapes: DynamicShapes = None,
    runtime_args: tuple[object, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run PyTorch and one requested ONNX export, then verify both against eager PyTorch.

    Static modes execute the example shape. Dynamic modes execute ``runtime_args`` when
    supplied, so the same pytest case checks that symbolic dimensions survive export. The ONNX
    graph must declare ``opset_version`` after conversion rather than silently retaining the
    exporter's native opset.
    """
    if mode.is_dynamic:
        assert dynamic_shapes is not None
        execution_args = runtime_args if runtime_args is not None else example_args
    else:
        execution_args = example_args

    model.eval()
    with torch.no_grad():
        expected = _flatten_tensor_output(model(*execution_args))
        selected_shapes = dynamic_shapes if mode.is_dynamic else None
        torch_program = torch.export.export(
            model,
            example_args,
            dynamic_shapes=selected_shapes,
        )
        torch_actual = _flatten_tensor_output(torch_program.module()(*execution_args))
        onnx_program = torch.onnx.export(
            model,
            example_args,
            dynamo=True,
            dynamic_shapes=selected_shapes,
            opset_version=opset_version,
        )
        assert isinstance(onnx_program, torch.onnx.ONNXProgram)
        _assert_onnx_shape_mode(onnx_program, mode)
        _assert_onnx_opset_version(onnx_program, opset_version)
        onnx_actual = _flatten_tensor_output(onnx_program(*execution_args))

    for actual in (torch_actual, onnx_actual):
        assert len(actual) == len(expected)
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)
    return onnx_actual
