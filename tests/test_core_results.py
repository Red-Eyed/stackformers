"""Expected core failures stay inspectable while public APIs keep their exceptions."""

from typing import TYPE_CHECKING

import pytest
import torch
from pydantic import ValidationError
from returns.result import Failure, Success

from stackformers.attention.cache_validation import (
    matching_layer_counts,
    single_token,
    supported_self_attention,
)
from stackformers.attention.cached import CachedSelfAttentionWrapper
from stackformers.attention.config import SelfAttentionConfig, _validate_attn_dims
from stackformers.attention.layout import matching_layouts
from stackformers.attention.self_attn import SelfAttention
from stackformers.decoder import Decoder, DecoderLayerBase
from stackformers.decoder_cache import CachedDecoderWrapper, _cached_layer, _unwrap_decoder
from stackformers.positional.config import RoPE1DConfig, RoPE2DConfig, RoPENDConfig, YaRNConfig
from stackformers.positional.none import NoPosEncoding
from stackformers.positional.rope1d import RotaryEmbedding1D
from stackformers.positional.rope2d import RotaryEmbedding2D
from stackformers.positional.validation import rotary_head_width
from stackformers.presets.decoder import TransformerDecoder, plain_decoder_config
from stackformers.presets.variable_width_encoder import (
    _validate_width_schedule,
    variable_width_encoder_config,
)
from stackformers.sequence import PaddedInput, make_padded_input, padded_to_packed

if TYPE_CHECKING:
    from stackformers.sequence import SequenceInput


@pytest.fixture
def padded_input() -> PaddedInput:
    """Provide native sequence records for joint layout validation."""
    return make_padded_input(torch.zeros(1, 2, 64), torch.ones(1, 2, dtype=torch.bool))


@pytest.fixture
def decoder() -> Decoder:
    """Provide a supported decoder whose cache admission can be exercised independently."""
    return TransformerDecoder(plain_decoder_config(dim=64, heads=1, num_layers=1))._decoder


@pytest.mark.parametrize("packed_queries", [False, True])
def test_layout_failure_is_a_shared_result(padded_input: PaddedInput, packed_queries: bool) -> None:
    """Both mismatched orders return inspectable errors without raising in the core."""
    packed = padded_to_packed(padded_input)
    queries: SequenceInput = packed if packed_queries else padded_input
    context: SequenceInput = padded_input if packed_queries else packed
    outcome = matching_layouts(queries, context)
    assert isinstance(outcome, Failure)
    assert isinstance(outcome.failure(), ValueError)
    assert str(outcome.failure()) == "cross-attention inputs must have matching layouts"


@pytest.mark.parametrize("packed", [False, True])
def test_layout_success_preserves_native_records(padded_input: PaddedInput, packed: bool) -> None:
    """Validation returns the original pair rather than copying tensors or metadata."""
    sequence: SequenceInput = padded_to_packed(padded_input) if packed else padded_input
    outcome = matching_layouts(sequence, sequence)
    assert isinstance(outcome, Success)
    queries, context = outcome.unwrap()
    assert queries is sequence
    assert context is sequence


@pytest.mark.parametrize(
    ("dim", "heads", "kv_heads", "message"),
    [
        (65, 2, None, "dim (65) must be divisible by heads (2)"),
        (64, 4, 3, "heads (4) must be divisible by kv_heads (3)"),
    ],
)
def test_head_geometry_failure_is_data_then_pydantic_error(
    dim: int,
    heads: int,
    kv_heads: int | None,
    message: str,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Core dimension failures are silent data; the public config retains ValidationError."""
    outcome = _validate_attn_dims(dim, heads, kv_heads)
    assert isinstance(outcome, Failure)
    assert str(outcome.failure()) == message
    assert len(recwarn) == 0
    with pytest.raises(ValidationError) as caught:
        SelfAttentionConfig(dim=dim, heads=heads, kv_heads=kv_heads)
    assert caught.value.errors()[0]["ctx"]["error"].args == (message,)


def test_yarn_ordering_failure_is_data_then_pydantic_error() -> None:
    """Bypass ingress only to test the pure check, then exercise real public admission."""
    config = YaRNConfig.model_construct(
        scale=2.0, original_max_seq_len=16, beta_fast=1.0, beta_slow=2.0
    )
    outcome = config._beta_ordering_result()
    assert isinstance(outcome, Failure)
    with pytest.raises(ValidationError) as caught:
        YaRNConfig(scale=2.0, original_max_seq_len=16, beta_fast=1.0, beta_slow=2.0)
    assert caught.value.errors()[0]["ctx"]["error"].args == outcome.failure().args


@pytest.mark.parametrize(("dim_head", "r_max"), [(6, 2.0), (4, 2.0), (8, 1.0)])
def test_rotary_geometry_failure_is_data_then_pydantic_error(dim_head: int, r_max: float) -> None:
    """Divisibility, band count, and spatial range failures retain boundary diagnostics."""
    config = RoPENDConfig.model_construct(dim_head=dim_head, coords=2, r_min=1.0, r_max=r_max)
    outcome = config._geometry_result()
    assert isinstance(outcome, Failure)
    with pytest.raises(ValidationError) as caught:
        RoPENDConfig(dim_head=dim_head, coords=2, r_min=1.0, r_max=r_max)
    assert caught.value.errors()[0]["ctx"]["error"].args == outcome.failure().args


@pytest.mark.parametrize(
    ("d_models", "dim_heads"), [([], []), ([64], []), ([0], [64]), ([65], [64])]
)
def test_width_schedule_failure_is_data_then_value_error(
    d_models: list[int],
    dim_heads: list[int],
) -> None:
    """The preset still raises the same validation message from its Result core."""
    outcome = _validate_width_schedule(d_models, dim_heads)
    assert isinstance(outcome, Failure)
    with pytest.raises(ValueError) as caught:
        variable_width_encoder_config(d_models=d_models, dim_heads=dim_heads)
    assert caught.value.args == outcome.failure().args


@pytest.mark.parametrize(("dim_head", "divisor"), [(3, 2), (6, 4)])
def test_rotary_width_keeps_constructor_assertion(dim_head: int, divisor: int) -> None:
    """The core carries AssertionError while the existing constructors still raise it."""
    match divisor:
        case 2:
            message = "dim_head must be even for RoPE"
        case _:
            message = "dim_head must be divisible by 4 for 2-D RoPE"
    outcome = rotary_head_width(dim_head, divisor, message)
    assert isinstance(outcome, Failure)
    with pytest.raises(AssertionError) as caught:
        match divisor:
            case 2:
                RotaryEmbedding1D(RoPE1DConfig(dim_head=dim_head))
            case _:
                RotaryEmbedding2D(RoPE2DConfig(dim_head=dim_head))
    assert caught.value.args == outcome.failure().args


@pytest.mark.parametrize(("causal", "window_size"), [(False, None), (True, 4)])
def test_unsupported_cache_equation_is_data_then_exception(
    causal: bool,
    window_size: int | None,
) -> None:
    """Cache admission reports unsupported equations before the wrapper applies policy."""
    attention = SelfAttention(
        SelfAttentionConfig(dim=64, heads=1, causal=causal, window_size=window_size),
        NoPosEncoding(),
    )
    outcome = supported_self_attention(attention)
    assert isinstance(outcome, Failure)
    with pytest.raises(NotImplementedError) as caught:
        CachedSelfAttentionWrapper(attention)
    assert caught.value.args == outcome.failure().args


@pytest.mark.parametrize("token_count", [0, 2])
def test_invalid_cached_token_count_is_data(token_count: int) -> None:
    """Unsupported step lengths return the existing diagnostic without unwinding."""
    outcome = single_token(token_count)
    assert isinstance(outcome, Failure)
    assert str(outcome.failure()) == "cached self-attention requires exactly one target token"


@pytest.mark.parametrize(
    ("cross_layers", "self_layers", "message"),
    [
        (1, 1, "cross cache has 1 layers, decoder has 2"),
        (2, 1, "self cache has 1 layers, decoder has 2"),
    ],
)
def test_cache_layer_failures_preserve_validation_order(
    cross_layers: int,
    self_layers: int,
    message: str,
) -> None:
    """Core validation selects the established first error when both caches mismatch."""
    outcome = matching_layer_counts(cross_layers, self_layers, 2)
    assert isinstance(outcome, Failure)
    assert str(outcome.failure()) == message


def test_decoder_resolution_failure_is_data_then_type_error() -> None:
    """An unsupported public module stays diagnosable through the core Result interface."""
    module = torch.nn.Identity()
    outcome = _unwrap_decoder(module)
    assert isinstance(outcome, Failure)
    with pytest.raises(TypeError) as caught:
        CachedDecoderWrapper(module)
    assert caught.value.args == outcome.failure().args


def test_cached_layer_captures_expected_constructor_failure(decoder: Decoder) -> None:
    """Building through a public constructor does not leak its expected cache rejection."""
    layer = decoder.layers[0]
    assert isinstance(layer, DecoderLayerBase)
    layer.self_attn = SelfAttention(SelfAttentionConfig(dim=64, heads=1), NoPosEncoding())
    outcome = _cached_layer(layer)
    assert isinstance(outcome, Failure)
    assert isinstance(outcome.failure(), NotImplementedError)
    with pytest.raises(NotImplementedError) as caught:
        CachedDecoderWrapper(decoder)
    assert caught.value.args == outcome.failure().args
