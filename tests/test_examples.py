"""Regression coverage for the small runnable examples."""

from __future__ import annotations

import math

from examples.causal_language_model import run_example as run_causal_language_model
from examples.image_encoder import run_example as run_image_encoder
from examples.padded_and_packed import run_example as run_padded_and_packed
from examples.point_cloud_encoder import run_example as run_point_cloud_encoder
from examples.subclassed_encoder import run_example as run_subclassed_encoder


def test_padded_and_packed_example_matches_valid_tokens() -> None:
    """Equivalent padded and packed batches produce the same valid-token output."""
    result = run_padded_and_packed()

    assert result.padded_shape == (2, 4, 64)
    assert result.packed_shape == (6, 64)
    assert result.max_abs_error < 1e-5


def test_causal_language_model_example_trains_one_step() -> None:
    """The causal language model produces finite loss, gradients, and an update."""
    result = run_causal_language_model()

    assert result.logits_shape == (2, 6, 32)
    assert math.isfinite(result.loss)
    assert result.loss > 0.0
    assert result.gradient_norm > 0.0
    assert result.parameter_update_norm > 0.0


def test_image_encoder_example_uses_the_complete_patch_grid() -> None:
    """The image encoder maps every patch to one fixed-size class prediction."""
    result = run_image_encoder()

    assert result.patch_tokens == 16
    assert result.logits_shape == (2, 10)


def test_point_cloud_example_is_translation_invariant() -> None:
    """Shifting every point by one vector preserves the point-cloud output."""
    result = run_point_cloud_encoder()

    assert result.output_shape == (2, 5, 192)
    assert result.translation_max_abs_error < 1e-5


def test_subclassed_encoder_example_uses_user_components() -> None:
    """Both preset extension levels construct and execute the custom branch."""
    result = run_subclassed_encoder()

    assert result.focused_output_shape == (2, 5, 64)
    assert result.custom_output_shape == (2, 5, 64)
    assert result.focused_custom_ff_layers == 2
    assert result.custom_custom_ff_layers == 2
