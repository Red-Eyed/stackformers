"""Run one Stackformers encoder-decoder eagerly and through cached ONNX Runtime graphs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers import (
    CachedDecoderWrapper,
    DecoderCrossAttentionCache,
    DecoderCrossAttentionCacheBuilder,
    PaddedInput,
    TransformerDecoder,
    TransformerEncoder,
    make_padded_input,
    plain_decoder_config,
    plain_encoder_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DIM = 128
HEADS = 2
LAYERS = 2
EXAMPLE_BATCH = 2
MAX_SOURCE_TOKENS = 6
MAX_TARGET_TOKENS = 4
CROSS_CACHE_NAMES = ("cross_kv_cache", "context_mask")


class ExampleResult(NamedTuple):
    """Files and parity measurements produced by :func:`run_example`."""

    encoder_path: Path
    cache_builder_path: Path
    cached_decoder_path: Path
    prefix_lengths: tuple[int, ...]
    max_abs_error: float


class EncoderModel(nn.Module):
    """Standalone source encoder that returns decoder-ready memory."""

    def __init__(self) -> None:
        """Build an export-friendly Stackformers encoder preset."""
        super().__init__()
        self.encoder = TransformerEncoder(plain_encoder_config(DIM, HEADS, LAYERS, ff_mult=3.0))

    @override
    def forward(self, source_input: PaddedInput) -> PaddedInput:
        """Encode source features while retaining their mask and absolute positions."""
        context = self.encoder(source_input)
        return source_input._replace(x=context)

    if TYPE_CHECKING:
        __call__ = forward


class EncoderDecoderModel(nn.Module):
    """Compose independently executable Stackformers encoder and decoder modules."""

    def __init__(self) -> None:
        """Build matching encoder and decoder presets with export-friendly dimensions."""
        super().__init__()
        self.encoder = EncoderModel()
        self.decoder = TransformerDecoder(plain_decoder_config(DIM, HEADS, LAYERS, ff_mult=3.0))

    @override
    def forward(self, source_input: PaddedInput, target_input: PaddedInput) -> Tensor:
        """Run the ordinary uncached encoder-decoder path."""
        memory = self.encoder(source_input)
        return self.decoder(target_input, memory)

    if TYPE_CHECKING:
        __call__ = forward


def _make_input(batch: int, tokens: int, *, seed: int) -> PaddedInput:
    """Create deterministic dense padded input for export examples and runtime checks."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, tokens, DIM, generator=generator)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return make_padded_input(x, mask)


def _padded_input_dynamic_shapes(
    module: nn.Module,
    input: PaddedInput,
    *,
    token_dim_name: str,
) -> dict[str, object]:
    """Describe dynamic batch and token dimensions for one padded-input graph."""
    batch = torch.export.Dim("batch", min=1, max=EXAMPLE_BATCH)
    tokens = torch.export.Dim(token_dim_name, min=1, max=MAX_SOURCE_TOKENS)
    shapes = torch.export.ShapesCollection()
    for tensor in input:
        shapes[tensor] = {0: batch, 1: tokens}
    dynamic_shapes: dict[str, object] = shapes.dynamic_shapes(module, (input,))
    return dynamic_shapes


def _cached_decoder_dynamic_shapes(
    module: nn.Module,
    target_input: PaddedInput,
    cross_cache: DecoderCrossAttentionCache,
    self_kv_cache: Tensor,
    step_i: Tensor,
) -> dict[str, object]:
    """Describe dynamic batch, context, and past dimensions for one-token decoding."""
    batch = torch.export.Dim("batch", min=1, max=EXAMPLE_BATCH)
    source_tokens = torch.export.Dim("source_tokens", min=1, max=MAX_SOURCE_TOKENS)
    past_tokens = torch.export.Dim("past_tokens", min=0, max=MAX_TARGET_TOKENS)
    shapes = torch.export.ShapesCollection()
    for tensor in target_input:
        shapes[tensor] = {0: batch}
    shapes[cross_cache.kv] = {2: batch, 4: source_tokens}
    shapes[cross_cache.context.mask] = {0: batch, 1: source_tokens}
    shapes[self_kv_cache] = {2: batch, 4: past_tokens}
    dynamic_shapes: dict[str, object] = shapes.dynamic_shapes(
        module,
        (target_input, cross_cache, self_kv_cache, step_i),
    )
    return dynamic_shapes


def _export_models(
    model: EncoderDecoderModel,
    source_input: PaddedInput,
    target_input: PaddedInput,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Export separate encoder, cache-construction, and repeated-decoder graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder.eval()
    cache_builder = DecoderCrossAttentionCacheBuilder(model.decoder).eval()
    cached_decoder = CachedDecoderWrapper(model.decoder).eval()
    with torch.no_grad():
        memory = encoder(source_input)
        cross_cache = cache_builder(memory)
    target_token = _token(target_input, 0)
    self_kv_cache = target_input.x.new_zeros(
        LAYERS,
        2,
        target_input.x.shape[0],
        model.decoder.config.self_attn.effective_kv_heads,
        MAX_TARGET_TOKENS - 2,
        model.decoder.config.self_attn.dim_head,
    )
    step_i = torch.tensor([MAX_TARGET_TOKENS - 2], dtype=torch.int64)

    encoder_program = torch.onnx.export(
        encoder,
        (source_input,),
        dynamo=True,
        dynamic_shapes=_padded_input_dynamic_shapes(
            encoder,
            source_input,
            token_dim_name="source_tokens",
        ),
        input_names=["source_x", "source_mask", "source_positions"],
        output_names=["memory_x", "memory_mask", "memory_positions"],
    )
    cache_program = torch.onnx.export(
        cache_builder,
        (memory,),
        dynamo=True,
        dynamic_shapes=_padded_input_dynamic_shapes(
            cache_builder,
            memory,
            token_dim_name="source_tokens",
        ),
        input_names=["memory_x", "memory_mask", "memory_positions"],
        output_names=list(CROSS_CACHE_NAMES),
    )
    decoder_program = torch.onnx.export(
        cached_decoder,
        (target_token, cross_cache, self_kv_cache, step_i),
        dynamo=True,
        dynamic_shapes=_cached_decoder_dynamic_shapes(
            cached_decoder,
            target_token,
            cross_cache,
            self_kv_cache,
            step_i,
        ),
        input_names=[
            "target_x",
            "target_mask",
            "target_positions",
            *CROSS_CACHE_NAMES,
            "self_kv_cache",
            "step_i",
        ],
        output_names=["decoder_output", "self_kv_cache_out"],
    )
    assert isinstance(encoder_program, torch.onnx.ONNXProgram)
    assert isinstance(cache_program, torch.onnx.ONNXProgram)
    assert isinstance(decoder_program, torch.onnx.ONNXProgram)
    encoder_path = output_dir / "encoder.onnx"
    cache_builder_path = output_dir / "decoder_cross_cache.onnx"
    cached_decoder_path = output_dir / "cached_decoder.onnx"
    encoder_program.save(encoder_path)
    cache_program.save(cache_builder_path)
    decoder_program.save(cached_decoder_path)
    return encoder_path, cache_builder_path, cached_decoder_path


def _numpy(tensor: Tensor) -> np.ndarray:
    """Detach one CPU tensor for an ONNX Runtime input."""
    return tensor.detach().numpy()


def _array_outputs(outputs: Sequence[object]) -> tuple[np.ndarray, ...]:
    """Validate tensor-valued ONNX Runtime outputs at the external-data boundary."""
    arrays: list[np.ndarray] = []
    for output in outputs:
        if not isinstance(output, np.ndarray):
            raise TypeError(f"expected ndarray output, received {type(output).__name__}")
        arrays.append(output)
    return tuple(arrays)


def _padded_feeds(prefix: str, input: PaddedInput) -> dict[str, np.ndarray]:
    """Flatten a padded input into the explicit names used by the ONNX graphs."""
    return {
        f"{prefix}_x": _numpy(input.x),
        f"{prefix}_mask": _numpy(input.mask),
        f"{prefix}_positions": _numpy(input.abs_positions),
    }


def _prefix(input: PaddedInput, tokens: int) -> PaddedInput:
    """Take one growing autoregressive prefix without rebuilding its metadata."""
    return PaddedInput(
        x=input.x[:, :tokens],
        mask=input.mask[:, :tokens],
        abs_positions=input.abs_positions[:, :tokens],
    )


def _token(input: PaddedInput, step: int) -> PaddedInput:
    """Select the one target token consumed by a cached decoder invocation."""
    return PaddedInput(
        x=input.x[:, step : step + 1],
        mask=input.mask[:, step : step + 1],
        abs_positions=input.abs_positions[:, step : step + 1],
    )


def _run_onnxruntime(
    model: EncoderDecoderModel,
    encoder_path: Path,
    cache_builder_path: Path,
    cached_decoder_path: Path,
) -> tuple[tuple[int, ...], float]:
    """Pass encoder memory through the decoder cache and verify growing prefixes."""
    source_input = _make_input(1, MAX_SOURCE_TOKENS - 1, seed=3)
    full_target = _make_input(1, MAX_TARGET_TOKENS - 1, seed=4)
    encoder_session = ort.InferenceSession(
        str(encoder_path),
        providers=["CPUExecutionProvider"],
    )
    cache_session = ort.InferenceSession(
        str(cache_builder_path),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(cached_decoder_path),
        providers=["CPUExecutionProvider"],
    )
    memory_names = ("memory_x", "memory_mask", "memory_positions")
    memory_values = _array_outputs(
        encoder_session.run(list(memory_names), _padded_feeds("source", source_input))
    )
    memory_feeds = dict(zip(memory_names, memory_values, strict=True))
    cache_values = _array_outputs(cache_session.run(list(CROSS_CACHE_NAMES), memory_feeds))
    cache_feeds = dict(zip(CROSS_CACHE_NAMES, cache_values, strict=True))
    self_kv_cache = np.zeros(
        (LAYERS, 2, 1, HEADS, 0, DIM // HEADS),
        dtype=np.float32,
    )
    prefix_lengths = tuple(range(1, full_target.x.shape[1] + 1))
    max_abs_error = 0.0

    with torch.no_grad():
        for step, prefix_length in enumerate(prefix_lengths):
            target_token = _token(full_target, step)
            expected = model(source_input, _prefix(full_target, prefix_length))[:, -1:]
            feeds = {
                **_padded_feeds("target", target_token),
                **cache_feeds,
                "self_kv_cache": self_kv_cache,
                "step_i": np.asarray([step], dtype=np.int64),
            }
            actual, self_kv_cache = _array_outputs(
                decoder_session.run(
                    ["decoder_output", "self_kv_cache_out"],
                    feeds,
                )
            )
            np.testing.assert_allclose(actual, _numpy(expected), rtol=1e-4, atol=1e-5)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(actual - _numpy(expected)))))

    return prefix_lengths, max_abs_error


def run_example(output_dir: Path) -> ExampleResult:
    """Export and execute all deployment graphs, returning paths and parity measurements."""
    torch.manual_seed(0)
    model = EncoderDecoderModel().eval()
    source_input = _make_input(EXAMPLE_BATCH, MAX_SOURCE_TOKENS, seed=1)
    target_input = _make_input(EXAMPLE_BATCH, MAX_TARGET_TOKENS, seed=2)
    with torch.no_grad():
        _ = model(source_input, target_input)
    encoder_path, cache_builder_path, cached_decoder_path = _export_models(
        model,
        source_input,
        target_input,
        output_dir,
    )
    prefix_lengths, max_abs_error = _run_onnxruntime(
        model,
        encoder_path,
        cache_builder_path,
        cached_decoder_path,
    )
    return ExampleResult(
        encoder_path=encoder_path,
        cache_builder_path=cache_builder_path,
        cached_decoder_path=cached_decoder_path,
        prefix_lengths=prefix_lengths,
        max_abs_error=max_abs_error,
    )


def main() -> None:
    """Run the demonstration in a temporary directory and print the verified result."""
    with TemporaryDirectory(prefix="stackformers-onnx-") as temp_dir:
        result = run_example(Path(temp_dir))
        print(f"encoder: {result.encoder_path}")
        print(f"cache builder: {result.cache_builder_path}")
        print(f"cached decoder: {result.cached_decoder_path}")
        print(f"decoded prefixes: {result.prefix_lengths}")
        print(f"maximum absolute error: {result.max_abs_error:.3e}")


if __name__ == "__main__":
    main()
