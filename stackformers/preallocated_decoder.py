"""Opt-in opset-24 decoder cache updates with explicitly owned, fixed-capacity buffers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from stackformers._result import unwrap_or_raise
from stackformers.all_valid_decoder import (
    LayerCaches,
    _admit_decoder,
    _allocate_self_cache,
    _decode_layer,
    _validate_layer_counts,
    _validate_token,
)
from stackformers.attention.all_valid import SelfAttentionModule, project_output
from stackformers.attention.cache import CrossAttentionKVCache
from stackformers.attention.cache_validation import supported_self_attention
from stackformers.attention.cached import _positioned_qkv
from stackformers.attention.onnx_backend import update_buffer, valid_length_attention

if TYPE_CHECKING:
    from jaxtyping import Float, Int

    from stackformers.decoder import DecoderLayerBase
    from stackformers.sequence import PaddedTensor


class PreallocatedDecoderCache(NamedTuple):
    """Exclusive per-sequence buffers and the number of initialized tokens.

    K/V tensors have shape (1, kv_heads, capacity, dim_head), one pair per layer.
    ``valid_length`` is an int64 tensor of shape (1,). Unused slots must be finite.
    Eager decoding mutates these buffers; retained snapshots alias later writes.
    Concurrent calls and independent decoding branches need separate buffers.
    """

    kv: LayerCaches
    valid_length: Int[Tensor, "b"]


class PreallocatedDecoderStepOutput(NamedTuple):
    """Decoder embeddings and the updated buffer references and valid length."""

    x: Float[Tensor, "b one d"]
    self_cache: PreallocatedDecoderCache


def _preallocated_self_attention(
    attention: SelfAttentionModule,
    input: PaddedTensor,
    cache: CrossAttentionKVCache,
    index: Int[Tensor, "b"],
) -> tuple[Tensor, CrossAttentionKVCache]:
    """Write only the new positioned K/raw V, then attend to the valid prefix."""
    q, k, v = _positioned_qkv(attention, input.x, input.abs_positions)
    updated_k = update_buffer(cache.k, k, index)
    updated_v = update_buffer(cache.v, v, index)
    output = valid_length_attention(q, updated_k, updated_v, index + 1)
    return project_output(attention, output), CrossAttentionKVCache(updated_k, updated_v)


class PreallocatedCachedDecoderWrapper(nn.Module):
    """Inference-only single-token decoder with an explicit mutable cache strategy.

    Requires PyTorch's ONNX symbolic API and ONNX opset >=24 for export. Each
    layer's K and V remain separate graph inputs/outputs, with no cache stacking.
    The ORT host must validate length before every call and bind corresponding
    cache inputs/outputs to the same compatible CPU buffers to request sharing.
    Eager execution checks overflow before mutation. Exported graphs rely on the
    host for that check; there is no automatic resize or circular overwrite.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Share admitted source layers without changing the ordinary decoder's state."""
        super().__init__()
        self.layers, self.final_norm = _admit_decoder(decoder)
        # Recursive train() would reset the source modules' individual modes.
        self.training = decoder.training
        self.layers.training = decoder.training

    def allocate_cache(self, capacity: int, *, dtype: torch.dtype) -> PreallocatedDecoderCache:
        """Allocate zero-filled buffers on the model device for the execution dtype.

        Use the autocast projection dtype when decoding under autocast. Allocation
        is once per sequence; resetting valid_length permits reusing finite slots.
        """
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        device = next(self.parameters()).device
        buffers = _allocate_self_cache(self.layers, capacity, dtype)
        return PreallocatedDecoderCache(buffers, torch.zeros(1, device=device, dtype=torch.int64))

    def validate_cache(self, cache: PreallocatedDecoderCache) -> None:
        """Check the host length/shape contract before execution, without writing buffers.

        Call this outside exported execution. ORT hosts must implement the same
        check: 0 <= valid_length < capacity, with consistent capacity per layer.
        """
        unwrap_or_raise(_validate_layer_counts(len(self.layers), cache.kv, None))
        if cache.valid_length.shape != (1,) or cache.valid_length.dtype != torch.int64:
            raise ValueError("valid_length must be an int64 tensor of shape (1,)")
        if cache.kv[0].k.ndim != 4:
            raise ValueError("cache buffers must have rank four")
        if cache.valid_length.device != next(self.parameters()).device:
            raise ValueError("valid_length must be on the model device")
        capacity = cache.kv[0].k.shape[2]
        length = int(cache.valid_length.item())
        if length < 0 or length >= capacity:
            raise ValueError("cache length must be nonnegative and below capacity before append")
        storage_pointers: set[int] = set()
        for layer, buffers in zip(self.layers, cache.kv, strict=True):
            source = cast("DecoderLayerBase", layer)
            attention = unwrap_or_raise(supported_self_attention(source.self_attn))
            config = attention.config
            shape = (1, config.effective_kv_heads, capacity, config.dim_head)
            parameter = next(attention.parameters())
            dtype = parameter.dtype
            if torch.is_autocast_enabled(parameter.device.type):
                dtype = torch.get_autocast_dtype(parameter.device.type)
            for buffer in buffers:
                if (
                    buffer.shape != shape
                    or buffer.dtype != dtype
                    or buffer.device != parameter.device
                ):
                    raise ValueError(
                        "cache buffers must match capacity, geometry, execution dtype and device"
                    )
                pointer = buffer.untyped_storage().data_ptr()
                if not buffer.is_contiguous() or pointer in storage_pointers:
                    raise ValueError("cache buffers must be contiguous and have disjoint storage")
                storage_pointers.add(pointer)

    @override
    @torch.no_grad()
    def forward(
        self,
        input: PaddedTensor,
        cross_cache: LayerCaches | None,
        self_cache: PreallocatedDecoderCache,
    ) -> PreallocatedDecoderStepOutput:
        """Append without resizing; return new length while retaining the mutable buffers."""
        if self.training:
            raise ValueError("preallocated decoding requires eval mode")
        unwrap_or_raise(_validate_token(input))
        unwrap_or_raise(_validate_layer_counts(len(self.layers), self_cache.kv, cross_cache))
        if not torch.compiler.is_compiling() and not torch.onnx.is_in_onnx_export():
            self.validate_cache(self_cache)
        updated: list[CrossAttentionKVCache] = []
        for index, layer in enumerate(self.layers):
            source = cast("DecoderLayerBase", layer)
            attention = cast("SelfAttentionModule", source.self_attn)
            self_call = partial(
                _preallocated_self_attention,
                attention,
                cache=self_cache.kv[index],
                index=self_cache.valid_length,
            )
            context = None if cross_cache is None else cross_cache[index]
            input, buffers = _decode_layer(source, input, self_call, context)
            updated.append(buffers)
        cache = PreallocatedDecoderCache(tuple(updated), self_cache.valid_length + 1)
        return PreallocatedDecoderStepOutput(self.final_norm(input.x), cache)

    if TYPE_CHECKING:
        __call__ = forward
