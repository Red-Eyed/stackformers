"""Regression checks for precise legacy-opset expected failures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from onnx import helper
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph
from onnxscript import ir

from tests.export_utils import (
    ONNX_OPSET_CASES,
    ExportShapeMode,
    _assert_onnx_opset_version,
    _run_onnx,
    export_and_run,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def onnx_program_factory() -> Callable[[str, int], torch.onnx.ONNXProgram]:
    """Build minimal real program objects without invoking an exporter or runtime."""

    def build(operator: str, opset: int) -> torch.onnx.ONNXProgram:
        """Create one operator with an explicit standard-domain opset."""
        node = helper.make_node(operator, ["x"], ["out"])
        graph = helper.make_graph([node], "test", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        return torch.onnx.ONNXProgram(ir.serde.deserialize_model(model), None)

    return build


def test_opset_parameters_do_not_blanket_xfail() -> None:
    """Supported cases must pass, and whole-test failures must never be suppressed."""
    assert all(not case.marks for case in ONNX_OPSET_CASES)


@pytest.mark.parametrize("opset", [15, 16, 17, 18, 25])
def test_matching_opset_passes(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram], opset: int
) -> None:
    """Even reduction graphs pass normally when the requested version is present."""
    _assert_onnx_opset_version(onnx_program_factory("ReduceMean", opset), opset)


@pytest.mark.parametrize("operator", ["ReduceMean", "ReduceL2", "ReduceMin"])
@pytest.mark.parametrize("opset", [15, 16, 17])
def test_legacy_reduction_downconversion_xfails(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    operator: str,
    opset: int,
) -> None:
    """Recognize the observed reduction down-converter retaining opset 18."""
    with pytest.raises(pytest.xfail.Exception, match="axes-input down-conversion"):
        _assert_onnx_opset_version(onnx_program_factory(operator, 18), opset)


@pytest.mark.parametrize(
    ("operator", "actual", "requested"),
    [("Identity", 18, 15), ("ReduceSum", 18, 15), ("ReduceMean", 19, 15), ("ReduceMean", 18, 19)],
)
def test_unrelated_opset_mismatch_fails(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    operator: str,
    actual: int,
    requested: int,
) -> None:
    """Do not hide unrelated version mismatches or failures at required opsets."""
    with pytest.raises(AssertionError):
        _assert_onnx_opset_version(onnx_program_factory(operator, actual), requested)


@pytest.mark.parametrize("opset", [15, 16, 17])
def test_legacy_split_attribute_xfails(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    monkeypatch: pytest.MonkeyPatch,
    opset: int,
) -> None:
    """Recognize only the observed legacy Split schema error at the runtime boundary."""
    error = InvalidGraph("Unrecognized attribute: num_outputs for operator Split")
    monkeypatch.setattr(torch.onnx.ONNXProgram, "__call__", Mock(side_effect=error))
    with pytest.raises(pytest.xfail.Exception, match="Split down-conversion"):
        _run_onnx(onnx_program_factory("Split", opset), (), opset)


@pytest.mark.parametrize(
    ("error", "opset"),
    [
        (InvalidGraph("Unrecognized attribute: num_outputs for operator Split"), 18),
        (InvalidGraph("Unrecognized attribute: unrelated for operator Split"), 15),
        (RuntimeError("Unrecognized attribute: num_outputs for operator Split"), 15),
        (AssertionError("numerical regression"), 15),
    ],
)
def test_unrelated_runtime_failures_propagate(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    opset: int,
) -> None:
    """Neither similar messages nor a legacy target may swallow unrelated failures."""
    monkeypatch.setattr(torch.onnx.ONNXProgram, "__call__", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        _run_onnx(onnx_program_factory("Identity", opset), (), opset)
    assert caught.value is error


def test_successful_legacy_runtime_passes(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful legacy runtime result remains a normal pass."""
    expected = torch.ones(2)
    monkeypatch.setattr(torch.onnx.ONNXProgram, "__call__", Mock(return_value=(expected,)))
    assert _run_onnx(onnx_program_factory("Identity", 15), (), 15)[0] is expected


def test_torch_parity_checked_before_onnx_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy ONNX failure cannot hide a wrong torch.export result."""
    torch_program = Mock()
    torch_program.module.return_value = Mock(return_value=torch.zeros(2))
    monkeypatch.setattr(torch.export, "export", Mock(return_value=torch_program))
    onnx_export = Mock(side_effect=AssertionError("ONNX must not run before torch parity"))
    monkeypatch.setattr(torch.onnx, "export", onnx_export)
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        export_and_run(torch.nn.Identity(), (torch.ones(2),), ExportShapeMode.STATIC, 15)
    onnx_export.assert_not_called()


def test_legacy_onnx_numerical_mismatch_fails(
    onnx_program_factory: Callable[[str, int], torch.onnx.ONNXProgram],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful legacy conversion still enforces ONNX numerical correctness."""
    program = onnx_program_factory("Identity", 15)
    monkeypatch.setattr(torch.onnx, "export", Mock(return_value=program))
    monkeypatch.setattr(torch.onnx.ONNXProgram, "__call__", Mock(return_value=(torch.zeros(2),)))
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        export_and_run(torch.nn.Identity(), (torch.ones(2),), ExportShapeMode.STATIC, 15)
