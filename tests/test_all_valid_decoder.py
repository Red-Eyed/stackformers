"""Deterministic all-valid decoding parity, admission, state and export regressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch.utils._pytree import tree_flatten

import stackformers as sf
from tests.conftest import atol

if TYPE_CHECKING:
    from collections.abc import Iterator

    from stackformers.all_valid_decoder import LayerCaches


@pytest.fixture(params=[False, True], ids=["ordinary", "packed"])
def packed(request: pytest.FixtureRequest) -> bool:
    """Exercise both weight layouts from the same deterministic original weights."""
    return bool(request.param)


@pytest.fixture(params=["pre", "post", "sandwich", "reordered"])
def topology(request: pytest.FixtureRequest) -> sf.NormPlacement:
    """Cover every built-in decoder residual equation."""
    return cast("sf.NormPlacement", request.param)


@pytest.fixture(params=[1, 2], ids=["gqa", "mha"])
def kv_heads(request: pytest.FixtureRequest) -> int:
    """Ensure caches retain KV-head width under grouped attention."""
    return int(request.param)


@pytest.fixture
def decoder(packed: bool, topology: sf.NormPlacement, kv_heads: int) -> sf.TransformerDecoder:
    """Build two layers with nontrivial QK normalization and rotary self/cross positions."""
    config = sf.plain_decoder_config(16, 2, 2, norm_placement=topology)
    config.self_attn = config.self_attn.model_copy(update={"qk_norm": True, "kv_heads": kv_heads})
    config.cross_attn = config.cross_attn.model_copy(update={"qk_norm": True, "kv_heads": kv_heads})
    config.ff = sf.HardSwishGLUConfig(dim=16, mult=3)
    model = sf.TransformerDecoder(config).eval()
    for layer in model._decoder.layers:
        attention = cast("sf.CrossAttention", cast("sf.DecoderLayerBase", layer).cross_attn)
        attention.pos_encoding = sf.RotaryEmbeddingND(
            sf.RoPENDConfig(dim_head=8, coords=1, r_min=1, r_max=32)
        )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            if parameter.ndim == 1:
                parameter.copy_(1 + 0.1 * torch.sin(values + index))
            else:
                parameter.copy_(0.1 * torch.sin(values * 0.17 + index))
    if packed:
        sf.to_packed(model)
    return model


@pytest.fixture
def memory() -> sf.AllValidInput:
    """Use nonzero, nonconsecutive context coordinates to expose lost cross RoPE."""
    x = torch.linspace(-0.8, 1.2, 80).reshape(1, 5, 16)
    positions = torch.tensor([0.5, 1.75, 3.0, 5.5, 8.0]).reshape(1, 5, 1)
    return sf.AllValidInput(x, positions)


@pytest.fixture
def target() -> sf.AllValidInput:
    """Four distinct target tokens with explicit absolute positions."""
    x = torch.cos(torch.arange(64).reshape(1, 4, 16) * 0.23)
    positions = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    return sf.AllValidInput(x, positions)


def _padded(input: sf.AllValidInput) -> sf.PaddedInput:
    """Represent the same values through the generic all-true padded contract."""
    mask = torch.ones(input.x.shape[:2], dtype=torch.bool, device=input.x.device)
    return sf.PaddedInput(input.x, mask, input.abs_positions)


def _token(input: sf.AllValidInput, index: int) -> sf.AllValidInput:
    """Select one token without resetting its absolute coordinates."""
    return sf.AllValidInput(
        input.x[:, index : index + 1], input.abs_positions[:, index : index + 1]
    )


def _prefix(input: sf.AllValidInput, length: int) -> sf.AllValidInput:
    """Retain a complete causal prefix for the independent reference path."""
    return sf.AllValidInput(input.x[:, :length], input.abs_positions[:, :length])


def _empty(cross: LayerCaches) -> LayerCaches:
    """Create independently owned empty self caches at projection dtype and KV width."""
    return tuple(
        sf.CrossAttentionKVCache(pair.k[:, :, :0].clone(), pair.v[:, :, :0].clone())
        for pair in cross
    )


@pytest.fixture
def inference() -> Iterator[None]:
    """Avoid building gradient history while comparing inference execution paths."""
    with torch.no_grad():
        yield


@pytest.fixture
def compilation_cache() -> Iterator[None]:
    """Isolate independently configured models from Dynamo's per-code recompile limit."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.usefixtures("inference")
def test_multi_step_parity(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    """Functional cached decoding matches padded/full-prefix paths without changing state."""
    device, dtype = device_dtype
    decoder.to(device=device, dtype=dtype)
    memory = sf.AllValidInput(
        memory.x.to(device=device, dtype=dtype), memory.abs_positions.to(device)
    )
    target = sf.AllValidInput(
        target.x.to(device=device, dtype=dtype), target.abs_positions.to(device)
    )
    state = {name: value.clone() for name, value in decoder.state_dict().items()}
    wrapper = sf.AllValidCachedDecoderWrapper(decoder)
    builder = sf.AllValidCrossAttentionCacheBuilder(decoder)
    assert {id(p) for p in wrapper.parameters()} == {id(p) for p in decoder.parameters()}
    cross = builder(memory)
    cross_before = tuple(sf.CrossAttentionKVCache(pair.k.clone(), pair.v.clone()) for pair in cross)
    cache = _empty(cross)
    for step in range(target.x.shape[1]):
        old = tuple(sf.CrossAttentionKVCache(pair.k.clone(), pair.v.clone()) for pair in cache)
        actual = wrapper(_token(target, step), cross, cache)
        expected = decoder(_padded(_prefix(target, step + 1)), _padded(memory))[:, -1:]
        full_valid = decoder(_prefix(target, step + 1), memory)[:, -1:]
        for output in (actual.x, full_valid):
            torch.testing.assert_close(output, expected, atol=atol(dtype), rtol=atol(dtype))
        torch.testing.assert_close(cache, old, rtol=0, atol=0)
        cache = actual.self_kv_cache
        assert cache[0].k.shape[2] == step + 1
    torch.testing.assert_close(cross, cross_before, rtol=0, atol=0)
    for name, value in decoder.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.usefixtures("inference")
def test_bfloat16_autocast(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
) -> None:
    """Both cache strategies retain bf16 projections under float32-weight autocast."""
    functional = sf.AllValidCachedDecoderWrapper(decoder)
    fixed = sf.PreallocatedCachedDecoderWrapper(decoder)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cross = sf.AllValidCrossAttentionCacheBuilder(decoder)(memory)
        cache = _empty(cross)
        preallocated = fixed.allocate_cache(4, dtype=torch.bfloat16)
        for step in range(4):
            token = _token(target, step)
            output = functional(token, cross, cache)
            updated = fixed(token, cross, preallocated)
            expected = decoder(_padded(_prefix(target, step + 1)), _padded(memory))[:, -1:]
            for actual in (output.x, updated.x):
                torch.testing.assert_close(
                    actual, expected, atol=atol(torch.bfloat16), rtol=atol(torch.bfloat16)
                )
            cache = output.self_kv_cache
            preallocated = updated.self_cache
            assert cache[0].k.dtype == torch.bfloat16


@pytest.mark.usefixtures("inference", "compilation_cache")
def test_context_free_export_and_compile(
    decoder: sf.TransformerDecoder,
    target: sf.AllValidInput,
) -> None:
    """Functional all-valid decoding remains traceable without cross-attention."""
    wrapper = sf.AllValidCachedDecoderWrapper(decoder)
    config = decoder.config.self_attn
    empty = target.x.new_zeros(1, config.effective_kv_heads, 0, config.dim_head)
    cache = tuple(sf.CrossAttentionKVCache(empty.clone(), empty.clone()) for _ in wrapper.layers)
    args = (_token(target, 0), None, cache)
    expected = wrapper(*args)
    exported = torch.export.export(wrapper, args).module()
    compiled = torch.compile(wrapper, backend="eager", fullgraph=True)
    torch.testing.assert_close(exported(*args), expected)
    torch.testing.assert_close(compiled(*args), expected)


def test_admission(decoder: sf.TransformerDecoder, target: sf.AllValidInput) -> None:
    """Reject multi-token calls and padded records before projection execution."""
    wrapper = sf.AllValidCachedDecoderWrapper(decoder)
    with pytest.raises(ValueError, match="exactly one token"):
        wrapper(target, None, ())
    with pytest.raises(TypeError, match="AllValidInput"):
        wrapper(_padded(_token(target, 0)), None, ())  # pyrefly: ignore [bad-argument-type]


@pytest.mark.parametrize("topology", ["pre"])
@pytest.mark.parametrize("kv_heads", [2])
def test_wrapper_construction_preserves_modes(decoder: sf.TransformerDecoder) -> None:
    """Creating execution views must not reset individual source module training modes."""
    layer = cast("sf.DecoderLayerBase", decoder._decoder.layers[0])
    cast("sf.CrossAttention", layer.cross_attn).dropout.train()
    modes = {name: module.training for name, module in decoder.named_modules()}
    sf.AllValidCachedDecoderWrapper(decoder)
    sf.AllValidCrossAttentionCacheBuilder(decoder)
    sf.PreallocatedCachedDecoderWrapper(decoder)
    assert {name: module.training for name, module in decoder.named_modules()} == modes


@pytest.mark.parametrize(
    "config", [sf.PackedSwiGLUConfig, sf.PackedGEGLUConfig, sf.PackedHardSwishGLUConfig]
)
@pytest.mark.parametrize("opset", [23, 24])
def test_packed_glu_graph(
    config: type[sf.PackedSwiGLUConfig]
    | type[sf.PackedGEGLUConfig]
    | type[sf.PackedHardSwishGLUConfig],
    opset: int,
) -> None:
    """Every packed GLU exports one shared input projection and its original output projection."""
    with torch.random.fork_rng():
        torch.manual_seed(31)
        module = cast("torch.nn.Module", sf.build_ff(config(dim=16, mult=3))).eval()
    x = torch.linspace(-1, 1, 48).reshape(1, 3, 16)
    with torch.no_grad():
        program = torch.onnx.export(module, (x,), dynamo=True, opset_version=opset)
        assert isinstance(program, torch.onnx.ONNXProgram)
        operations = [node.op_type for node in program.model_proto.graph.node]
        assert operations.count("MatMul") == 2
        assert operations.count("Split") == 1
        torch.testing.assert_close(program(x)[0], module(x))


@pytest.mark.parametrize("topology", ["pre"])
@pytest.mark.parametrize("kv_heads", [2])
@pytest.mark.usefixtures("inference")
def test_integrated_conversion(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
) -> None:
    """Conversion retains shared wrapper references and restores exact source checkpoints."""
    sf.to_unpacked(decoder)
    state = {name: tensor.clone() for name, tensor in decoder.state_dict().items()}
    functional = sf.AllValidCachedDecoderWrapper(decoder)
    fixed = sf.PreallocatedCachedDecoderWrapper(decoder)
    builder = sf.AllValidCrossAttentionCacheBuilder(decoder)
    tree = torch.nn.ModuleList([decoder, functional, fixed, builder])
    expected = decoder(_padded(_prefix(target, 1)), _padded(memory))
    for convert in (sf.to_packed, sf.to_unpacked):
        convert(tree)
        cross = builder(memory)
        actual = functional(_token(target, 0), cross, _empty(cross))
        updated = fixed(_token(target, 0), cross, fixed.allocate_cache(4, dtype=torch.float32))
        torch.testing.assert_close(actual.x, expected)
        torch.testing.assert_close(updated.x, expected)
        assert functional.layers[0] is decoder._decoder.layers[0]
        assert fixed.layers[0] is decoder._decoder.layers[0]
        assert builder.layers[0] is decoder._decoder.layers[0].cross_attn
    for name, tensor in decoder.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("topology", ["sandwich"])
@pytest.mark.parametrize("opset", [23, 24])
@pytest.mark.usefixtures("inference")
def test_onnx_dynamic_past(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    target: sf.AllValidInput,
    packed: bool,
    opset: int,
) -> None:
    """One graph runs empty/growing caches and keeps packed projections mask-free."""
    wrapper = sf.AllValidCachedDecoderWrapper(decoder)
    cross = sf.AllValidCrossAttentionCacheBuilder(decoder)(memory)
    cache = _empty(cross)
    for step in range(2):
        cache = wrapper(_token(target, step), cross, cache).self_kv_cache
    args = (_token(target, 2), cross, cache)
    shapes = torch.export.ShapesCollection()
    for tensor in tree_flatten(args[:2])[0]:
        shapes[tensor] = {}
    past = torch.export.Dim("past", min=0, max=8)
    for pair in cache:
        shapes[pair.k] = {2: past}
        shapes[pair.v] = {2: past}
    program = torch.onnx.export(
        wrapper,
        args,
        dynamo=True,
        opset_version=opset,
        dynamic_shapes=shapes.dynamic_shapes(wrapper, args),
    )
    assert isinstance(program, torch.onnx.ONNXProgram)
    model = program.model_proto
    onnx.checker.check_model(model)
    assert next(op.version for op in model.opset_import if op.domain == "") == opset
    operations = [node.op_type for node in model.graph.node]
    assert not {"IsInf", "Less", "LessOrEqual", "Where", "ReduceMin"}.intersection(operations)
    assert operations.count("MatMul") == (6 if packed else 9) * len(cross)
    assert operations.count("Split") == (2 * len(cross) if packed else 0)
    for node in model.graph.node:
        if node.op_type == "Attention":
            assert not node.input[3:4] or node.input[3] == ""
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attributes.get("is_causal", 0) == 0
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    cache = _empty(cross)
    for step in range(4):
        args = (_token(target, step), cross, cache)
        leaves, _ = tree_flatten(args)
        feeds = {
            v.name: tensor.numpy() for v, tensor in zip(session.get_inputs(), leaves, strict=True)
        }
        outputs = session.run(None, feeds)
        expected = decoder(_padded(_prefix(target, step + 1)), _padded(memory))[:, -1:]
        assert isinstance(outputs[0], np.ndarray)
        np.testing.assert_allclose(outputs[0], expected.numpy(), atol=1e-5, rtol=1e-4)
        cache = tuple(
            sf.CrossAttentionKVCache(
                torch.from_numpy(outputs[1 + 2 * i]), torch.from_numpy(outputs[2 + 2 * i])
            )
            for i in range(len(cross))
        )


@pytest.mark.parametrize("topology", ["pre"])
@pytest.mark.parametrize("kv_heads", [2])
@pytest.mark.parametrize("opset", [23, 24])
@pytest.mark.usefixtures("inference")
def test_encoder_and_cross_builder_export(
    decoder: sf.TransformerDecoder,
    memory: sf.AllValidInput,
    packed: bool,
    opset: int,
) -> None:
    """Encoder QKV/GLU and once-per-context KV projections retain packed MatMuls."""
    with torch.random.fork_rng():
        torch.manual_seed(19)
        encoder = sf.TransformerEncoder(sf.plain_encoder_config(16, 2, 1)).eval()
    if packed:
        sf.to_packed(encoder)
    builder = sf.AllValidCrossAttentionCacheBuilder(decoder)
    for module, expected_matmuls in (
        (encoder, 4 if packed else 7),
        (builder, len(builder.layers) * (1 if packed else 2)),
    ):
        program = torch.onnx.export(module, (memory,), dynamo=True, opset_version=opset)
        assert isinstance(program, torch.onnx.ONNXProgram)
        model = program.model_proto
        operations = [node.op_type for node in model.graph.node]
        assert operations.count("MatMul") == expected_matmuls
        assert not {"IsInf", "Less", "Where", "ReduceMin"}.intersection(operations)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(
            None,
            {
                v.name: tensor.numpy()
                for v, tensor in zip(session.get_inputs(), memory, strict=True)
            },
        )
        expected, _ = tree_flatten(module(memory))
        for actual, reference in zip(outputs, expected, strict=True):
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-4, atol=1e-5)
