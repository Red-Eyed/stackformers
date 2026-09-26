"""Attention configuration, layout, and packed-memory boundary regressions."""

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from torch import Tensor

from stackformers.attention import self_attn
from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.attention.cross_attn import CrossAttention
from stackformers.attention.layout import call_cross_attention
from stackformers.attention.self_attn import SelfAttention
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import PackedInput, PaddedInput, make_padded_input, padded_to_packed

if TYPE_CHECKING:
    from stackformers.sequence import SequenceInput


@pytest.fixture
def padded_input() -> PaddedInput:
    """Provide uneven document lengths to expose unnecessary repadding."""
    return make_padded_input(
        torch.randn(2, 3, 64), torch.tensor([[True, True, True], [True, False, False]])
    )


@pytest.mark.parametrize("config_type", [SelfAttentionConfig, CrossAttentionConfig])
def test_attention_config_is_frozen(
    config_type: type[SelfAttentionConfig] | type[CrossAttentionConfig],
) -> None:
    """Reject architecture mutation before it can reinterpret already-built projections."""
    config = config_type(dim=64, heads=1)
    with pytest.raises(ValidationError, match="frozen"):
        # Deliberately violate the static contract to verify runtime enforcement too.
        config.heads = 2  # pyrefly: ignore[read-only]
    assert config.heads == 1


@pytest.mark.parametrize("packed_queries", [False, True])
def test_mixed_layouts_are_rejected(padded_input: PaddedInput, packed_queries: bool) -> None:
    """Both mixed pairs fail clearly at concrete and protocol-dispatch boundaries."""
    attention = CrossAttention(CrossAttentionConfig(dim=64, heads=1), NoPosEncoding())
    packed = padded_to_packed(padded_input)
    queries: SequenceInput = packed if packed_queries else padded_input
    context: SequenceInput = padded_input if packed_queries else packed
    with pytest.raises(ValueError, match="matching layouts"):
        # Deliberately enter through the untyped framework boundary to test bad runtime input.
        nn.Module.__call__(attention, queries, context)
    with pytest.raises(ValueError, match="matching layouts"):
        call_cross_attention(attention, queries, context)


def test_no_bias_does_not_repad_features(
    padded_input: PaddedInput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-bias path never materializes padded features merely to calculate no bias."""
    attention = SelfAttention(SelfAttentionConfig(dim=64, heads=1), NoPosEncoding())
    expected = attention(padded_input)[padded_input.mask]

    def reject_repadding(input: PackedInput) -> PaddedInput:
        """Fail at the feature-allocation boundary, independently of attention backend."""
        raise AssertionError("no-bias attention must not repad input features")

    monkeypatch.setattr(self_attn, "packed_to_padded", reject_repadding)
    actual: Tensor = attention(padded_to_packed(padded_input))
    torch.testing.assert_close(actual, expected)
