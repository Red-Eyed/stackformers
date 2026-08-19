"""Shared execution and parity checks for PyTorch and ONNX exports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from onnxscript import ir
from torch.utils import _pytree

DynamicShapes: TypeAlias = dict[str, Any] | tuple[Any, ...] | list[Any] | None


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


def _flatten_tensor_output(output: Any) -> tuple[torch.Tensor, ...]:
    """Flatten an exported output tree and require tensor-only leaves."""
    leaves, _ = _pytree.tree_flatten(output)
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    return tuple(leaves)


def export_and_run(
    model: nn.Module,
    example_args: tuple[Any, ...],
    mode: ExportShapeMode,
    *,
    dynamic_shapes: DynamicShapes = None,
    runtime_args: tuple[Any, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run PyTorch and ONNX exports and verify both against eager PyTorch.

    Static modes execute the example shape. Dynamic modes execute ``runtime_args`` when
    supplied, so the same pytest case checks that symbolic dimensions survive export.
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
        )
        assert isinstance(onnx_program, torch.onnx.ONNXProgram)
        _assert_onnx_shape_mode(onnx_program, mode)
        onnx_actual = _flatten_tensor_output(onnx_program(*execution_args))

    for actual in (torch_actual, onnx_actual):
        assert len(actual) == len(expected)
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)
    return onnx_actual
