"""Fixed-capacity cache parity, overflow, ownership, and ONNX Runtime buffer sharing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch.utils._pytree import tree_flatten

import stackformers as sf
from tests.conftest import atol
from tests.test_all_valid_decoder import (
    _padded,
    _prefix,
    _token,
)
from tests.test_all_valid_decoder import (
    compilation_cache as compilation_cache,
)
from tests.test_all_valid_decoder import (
    decoder as decoder,
)
from tests.test_all_valid_decoder import (
    inference as inference,
)
from tests.test_all_valid_decoder import (
    kv_heads as kv_heads,
)
from tests.test_all_valid_decoder import (
    memory as memory,
)
from tests.test_all_valid_decoder import (
    packed as packed,
)
from tests.test_all_valid_decoder import (
    target as target,
)
from tests.test_all_valid_decoder import (
    topology as topology,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.usefixtures("inference")
def test_preallocated_steps(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Append from length zero through capacity without changing storage or model state."""
    device, dtype = device_dtype
    decoder.to(device=device, dtype=dtype)
    memory = sf.AllValidInput(
        memory.x.to(device=device, dtype=dtype), memory.abs_positions.to(device)
    )
    target = sf.AllValidInput(
        target.x.to(device=device, dtype=dtype), target.abs_positions.to(device)
    )
    state = {name: value.clone() for name, value in decoder.state_dict().items()}
    wrapper = sf.PreallocatedCachedDecoderWrapper(decoder)
    cross = sf.AllValidCrossAttentionCacheBuilder(decoder)(memory)
    cache = wrapper.allocate_cache(4, dtype=dtype)
    pointers = [t.data_ptr() for pair in cache.kv for t in pair]
    for step in range(4):
        output = wrapper(_token(target, step), cross, cache)
        expected = decoder(_padded(_prefix(target, step + 1)), _padded(memory))[:, -1:]
        torch.testing.assert_close(output.x, expected, atol=atol(dtype), rtol=atol(dtype))
        cache = output.self_cache
        assert cache.valid_length.item() == step + 1
        assert [t.data_ptr() for pair in cache.kv for t in pair] == pointers
        for pair in cache.kv:
            for tensor in pair:
                torch.testing.assert_close(
                    tensor[:, :, step + 1 :],
                    torch.zeros_like(tensor[:, :, step + 1 :]),
                    rtol=0,
                    atol=0,
                )
    before = tuple(pair.k.clone() for pair in cache.kv)
    with pytest.raises(ValueError, match="below capacity"):
        wrapper(_token(target, 0), cross, cache)
    for pair, unchanged in zip(cache.kv, before, strict=True):
        torch.testing.assert_close(pair.k, unchanged, atol=0, rtol=0)
    for name, value in decoder.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.usefixtures("inference")
def test_reject_aliases_and_wrong_dtype(decoder: sf.TransformerDecoder) -> None:
    """Reject unsafe buffer storage and execution dtypes before writing any layer."""
    wrapper = sf.PreallocatedCachedDecoderWrapper(decoder)
    cache = wrapper.allocate_cache(4, dtype=torch.float32)
    pair = sf.CrossAttentionKVCache(cache.kv[0].k, cache.kv[0].k)
    with pytest.raises(ValueError, match="disjoint"):
        wrapper.validate_cache(cache._replace(kv=(pair, *cache.kv[1:])))
    with pytest.raises(ValueError, match="dtype"):
        wrapper.validate_cache(wrapper.allocate_cache(4, dtype=torch.float16))
    with pytest.raises(ValueError, match="positive"):
        wrapper.allocate_cache(0, dtype=torch.float32)


@pytest.mark.usefixtures("inference", "compilation_cache")
def test_preallocated_export_and_compile(
    decoder: sf.TransformerDecoder,
    target: sf.AllValidInput,
) -> None:
    """PyTorch capture preserves one-token outputs and the explicit mutable cache contract."""
    wrapper = sf.PreallocatedCachedDecoderWrapper(decoder)
    cache = wrapper.allocate_cache(4, dtype=torch.float32)
    args = (_token(target, 0), None, cache)
    expected = wrapper(*args)
    exported = torch.export.export(wrapper, args).module()
    compiled = torch.compile(wrapper, backend="eager", fullgraph=True)
    torch.testing.assert_close(exported(*args), expected)
    torch.testing.assert_close(compiled(*args), expected)


@pytest.mark.parametrize("topology", ["sandwich"])
@pytest.mark.usefixtures("inference")
def test_onnx24_shared_buffers(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
    tmp_path: Path,
) -> None:
    """Full CPU decoder aliases cache I/O and preserves prefixes over multiple runs."""
    wrapper = sf.PreallocatedCachedDecoderWrapper(decoder)
    cross = sf.AllValidCrossAttentionCacheBuilder(decoder)(memory)
    cache = wrapper.allocate_cache(4, dtype=torch.float32)
    args = (_token(target, 0), cross, cache)
    cross_names = [f"cross_{i}_{kind}" for i in range(len(cross)) for kind in ("k", "v")]
    past_names = [f"past_{i}_{kind}" for i in range(len(cross)) for kind in ("k", "v")]
    present_names = [name.replace("past", "present") for name in past_names]
    names = ["token", "positions", *cross_names, *past_names, "length"]
    program = torch.onnx.export(
        wrapper,
        args,
        dynamo=True,
        opset_version=24,
        input_names=names,
        output_names=["output", *present_names, "next_length"],
    )
    assert isinstance(program, torch.onnx.ONNXProgram)
    model = program.model_proto
    onnx.checker.check_model(model)
    assert next(op.version for op in model.opset_import if op.domain == "") == 24
    operations = [node.op_type for node in model.graph.node]
    assert operations.count("TensorScatter") == 2 * len(cross)
    assert not {"IsInf", "Less", "LessOrEqual", "Where", "ReduceMin"}.intersection(operations)
    # Cache outputs must come directly from their scatter, without stacking or reshaping.
    producers = {output: node for node in model.graph.node for output in node.output}
    for past, present in zip(past_names, present_names, strict=True):
        assert producers[present].op_type == "TensorScatter"
        assert producers[present].input[0] == past
    for node in model.graph.node:
        if node.op_type == "Attention":
            assert not node.input[3:4] or node.input[3] == ""
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attributes.get("is_causal", 0) == 0
    program.save(tmp_path / "preallocated.onnx")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    leaves, _ = tree_flatten(args)
    values = {
        name: tensor.detach().numpy().copy() for name, tensor in zip(names, leaves, strict=True)
    }
    buffers = {name: ort.OrtValue.ortvalue_from_numpy(values[name]) for name in past_names}
    binding = session.io_binding()
    for name in cross_names:
        binding.bind_cpu_input(name, values[name])
    for past, present in zip(past_names, present_names, strict=True):
        binding.bind_ortvalue_input(past, buffers[past])
        binding.bind_ortvalue_output(present, buffers[past])
    binding.bind_output("output", "cpu")
    binding.bind_output("next_length", "cpu")
    for step in range(4):
        token = _token(target, step)
        binding.bind_cpu_input("token", token.x.numpy())
        binding.bind_cpu_input("positions", token.abs_positions.numpy())
        binding.bind_cpu_input("length", np.asarray([step], dtype=np.int64))
        session.run_with_iobinding(binding)
        outputs = dict(
            zip([*present_names, "output", "next_length"], binding.get_outputs(), strict=True)
        )
        expected = decoder(_padded(_prefix(target, step + 1)), _padded(memory))[:, -1:]
        np.testing.assert_allclose(
            outputs["output"].numpy(), expected.numpy(), rtol=1e-4, atol=1e-5
        )
        assert outputs["next_length"].numpy().item() == step + 1
        for past, present in zip(past_names, present_names, strict=True):
            assert outputs[present].data_ptr() == buffers[past].data_ptr()
            unused = values[past][:, :, step + 1 :]
            np.testing.assert_array_equal(unused, np.zeros_like(unused))
