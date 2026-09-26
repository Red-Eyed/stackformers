"""Tensor-only records for externally managed attention key/value caches."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from stackformers.sequence import PaddedSequence


class CrossAttentionKVCache(NamedTuple):
    """Projected context keys and values for one cross-attention layer.

    Keys and values retain the configured KV-head count. Query-head expansion stays in the
    attention operation so GQA and MQA caches do not consume full multi-head storage.
    """

    k: Float[Tensor, "b kv_h s dh"]
    v: Float[Tensor, "b kv_h s dh"]


class DecoderCrossAttentionCache(NamedTuple):
    """Dense context cache and validity mask shared by every decoder step.

    The leading dimensions of ``kv`` are layer and K/V selector. Keeping every layer in one
    tensor gives ONNX Runtime C++ one required ``Ort::Value`` instead of a variable-length list.
    """

    kv: Float[Tensor, "layers two b kv_h s dh"]
    context: PaddedSequence


class DecoderStepOutput(NamedTuple):
    """One-token decoder result and the one-position-longer self-attention cache."""

    x: Float[Tensor, "b one d"]
    self_kv_cache: Float[Tensor, "layers two b kv_h next_tokens dh"]
