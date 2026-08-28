"""Regression coverage for the runnable encoder-decoder ONNX Runtime example."""

from pathlib import Path

import onnxruntime as ort

from examples.encoder_decoder_onnxruntime import run_example


def test_encoder_decoder_onnxruntime_example(tmp_path: Path) -> None:
    """Separate encoder and cached-decoder graphs match eager growing-prefix output."""
    result = run_example(tmp_path)

    assert result.encoder_path.is_file()
    assert result.cache_builder_path.is_file()
    assert result.cached_decoder_path.is_file()
    assert result.prefix_lengths == (1, 2, 3)
    assert result.max_abs_error < 1e-4

    encoder_session = ort.InferenceSession(str(result.encoder_path))
    cache_session = ort.InferenceSession(str(result.cache_builder_path))
    decoder_session = ort.InferenceSession(str(result.cached_decoder_path))
    assert [input.name for input in encoder_session.get_inputs()] == [
        "source_x",
        "source_mask",
        "source_positions",
    ]
    assert [output.name for output in encoder_session.get_outputs()] == [
        "memory_x",
        "memory_mask",
        "memory_positions",
    ]
    assert [input.name for input in cache_session.get_inputs()] == [
        "memory_x",
        "memory_mask",
        "memory_positions",
    ]
    assert [input.name for input in decoder_session.get_inputs()] == [
        "target_x",
        "target_mask",
        "target_positions",
        "cross_kv_cache",
        "context_mask",
        "self_kv_cache",
        "step_i",
    ]
    assert [output.name for output in cache_session.get_outputs()] == [
        "cross_kv_cache",
        "context_mask",
    ]
    assert [output.name for output in decoder_session.get_outputs()] == [
        "decoder_output",
        "self_kv_cache_out",
    ]

    encoder_inputs = {input.name: input.shape for input in encoder_session.get_inputs()}
    for shape in encoder_inputs.values():
        assert isinstance(shape[0], str)
        assert isinstance(shape[1], str)

    cache_inputs = {input.name: input.shape for input in cache_session.get_inputs()}
    for shape in cache_inputs.values():
        assert isinstance(shape[0], str)
        assert isinstance(shape[1], str)

    decoder_inputs = {input.name: input.shape for input in decoder_session.get_inputs()}
    assert isinstance(decoder_inputs["target_x"][0], str)
    assert decoder_inputs["target_x"][1:] == [1, 128]
    assert isinstance(decoder_inputs["target_mask"][0], str)
    assert decoder_inputs["target_mask"][1:] == [1]
    assert isinstance(decoder_inputs["target_positions"][0], str)
    assert decoder_inputs["target_positions"][1:] == [1, 1]
    assert isinstance(decoder_inputs["cross_kv_cache"][2], str)
    assert isinstance(decoder_inputs["cross_kv_cache"][4], str)
    assert isinstance(decoder_inputs["context_mask"][0], str)
    assert isinstance(decoder_inputs["context_mask"][1], str)
    assert isinstance(decoder_inputs["self_kv_cache"][2], str)
    assert isinstance(decoder_inputs["self_kv_cache"][4], str)
    assert decoder_inputs["step_i"] == [1]
