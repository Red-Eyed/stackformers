from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stackformers.attention.config import SelfAttentionConfig
from stackformers.attention.self_attn import SelfAttention
from stackformers.encoder import Encoder
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from stackformers.layers import TransformerLayer
from stackformers.mlm.config import MLMWrapperConfig
from stackformers.mlm.wrapper import MLMOutput, MLMWrapper
from stackformers.norm.config import RMSNormConfig
from stackformers.norm.factory import build_norm
from stackformers.positional.none import NoPosEncoding
from stackformers.sequence import (
    PackedInput,
    PaddedInput,
    SequenceInput,
    make_packed_input,
    make_padded_input,
)

B, N, D, H, DH = 2, 16, 16, 4, 4
NUM_LAYERS = 2
NT = 10  # two packed seqs: 6 + 4


class AllMasking(nn.Module):
    """Test double: marks every valid position for masking."""

    def forward(self, input: SequenceInput) -> torch.Tensor:
        match input:
            case PaddedInput(mask=mask):
                return mask.clone()
            case PackedInput(x=x):
                return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)


def _build_encoder(device: torch.device, dtype: torch.dtype) -> Encoder:
    attn_cfg = SelfAttentionConfig(dim=D, heads=H, dim_head=DH)
    ff_cfg = SwiGLUConfig(dim=D)
    norm_cfg = RMSNormConfig(dim=D)
    layers = [
        TransformerLayer(
            self_attn=SelfAttention(attn_cfg, NoPosEncoding()),
            ff=SwiGLU(ff_cfg),
            norm_attn=build_norm(norm_cfg),
            norm_ff=build_norm(norm_cfg),
        )
        for _ in range(NUM_LAYERS)
    ]
    return Encoder(layers=layers, final_norm=build_norm(norm_cfg)).to(device=device, dtype=dtype)


@pytest.fixture
def config() -> MLMWrapperConfig:
    return MLMWrapperConfig(dim=D, mask_ratio=0.5)


@pytest.fixture
def x_pad(device_dtype: tuple[torch.device, torch.dtype]) -> PaddedInput:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    return make_padded_input(x, mask)


@pytest.fixture
def x_packed(device_dtype: tuple[torch.device, torch.dtype]) -> PackedInput:
    device, dtype = device_dtype
    if not device.type == "cuda" or dtype not in (torch.float16, torch.bfloat16):
        pytest.skip("packed attention requires CUDA with float16 or bfloat16")
    x = torch.randn(NT, D, device=device, dtype=dtype)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    return make_packed_input(x, cu, max_seqlen=6)


def test_wrapper_padded_output_shapes(
    config: MLMWrapperConfig,
    x_pad: PaddedInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    encoder = _build_encoder(device, dtype)
    wrapper = MLMWrapper(config).to(device=device, dtype=dtype)
    res = wrapper(x_pad, encoder)
    assert isinstance(res, MLMOutput)
    assert res.mlm_loss.shape == ()
    assert res.out.shape == (B, N, D)


def test_wrapper_packed_output_shapes(
    config: MLMWrapperConfig,
    x_packed: PackedInput,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    encoder = _build_encoder(device, dtype)
    wrapper = MLMWrapper(config).to(device=device, dtype=dtype)
    res = wrapper(x_packed, encoder)
    assert res.mlm_loss.shape == ()
    assert res.out.shape == (NT, D)


def test_wrapper_mask_token_receives_gradient(device: torch.device) -> None:
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config).to(device)
    x = torch.randn(B, N, D, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    res = wrapper(make_padded_input(x, mask), encoder)
    res.mlm_loss.backward()
    assert wrapper.mask_token.grad is not None


def test_wrapper_does_not_own_encoder(device: torch.device) -> None:
    """MLMWrapper takes the encoder at call time, not at construction — its own
    parameter tree must never include the encoder's weights, since the encoder is
    owned and registered wherever the caller already keeps it.
    """
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config).to(device)
    encoder_param_ids = {id(p) for p in encoder.parameters()}
    wrapper_param_ids = {id(p) for p in wrapper.parameters()}
    assert encoder_param_ids.isdisjoint(wrapper_param_ids)


def test_wrapper_detaches_target_from_upstream_input(device: torch.device) -> None:
    """Collapse guard (design doc §5): a trainable tokenizer upstream of x must get no
    gradient from this loss, or the optimizer can collapse every token to one constant
    vector to make reconstruction trivial.

    Force every position masked, so torch.where's backward zeroes x's contribution
    through the corrupted-x branch entirely (its condition is True everywhere, so the
    "x" branch of the select is never taken). The only remaining path from loss back to
    x would be through the target — and that's detached — so x's accumulated gradient
    must be exactly zero.
    """
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config, masking_strategy=AllMasking()).to(device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    res = wrapper(make_padded_input(x, mask), encoder)
    res.mlm_loss.backward()
    assert x.grad is not None
    assert torch.all(x.grad == 0)


def test_wrapper_accepts_custom_masking_strategy(device: torch.device) -> None:
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config, masking_strategy=AllMasking()).to(device)
    x = torch.randn(B, N, D, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    res = wrapper(make_padded_input(x, mask), encoder)
    assert res.mlm_loss.shape == ()


def test_wrapper_training_out_is_always_clean(device: torch.device) -> None:
    """out must never carry masking, even in training — only mlm_loss does. This keeps
    the main pipeline byte-identical whether or not the MLM aux loss trains alongside
    it, so callers never need to special-case which pass produced `out`. AllMasking
    maximizes the contrast: if masking leaked into `out`, every token would be replaced
    by mask_token, making it maximally different from a clean encoder(input) call.
    """
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config, masking_strategy=AllMasking()).to(device)
    x = torch.randn(B, N, D, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    input = make_padded_input(x, mask)

    res = wrapper(input, encoder)
    expected = encoder(input)

    assert torch.equal(res.out, expected)


def test_wrapper_eval_mode_passes_through_encoder_unmasked(device: torch.device) -> None:
    """In eval mode, forward() must skip masking entirely: out matches a clean encoder
    call exactly, and mlm_loss reports zero — so callers can invoke the wrapper
    unconditionally in both modes without an if-training branch of their own.
    """
    config = MLMWrapperConfig(dim=D, mask_ratio=0.5)
    encoder = _build_encoder(device, torch.float32)
    wrapper = MLMWrapper(config).to(device)
    wrapper.eval()
    x = torch.randn(B, N, D, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    input = make_padded_input(x, mask)

    res = wrapper(input, encoder)
    expected = encoder(input)

    assert torch.equal(res.out, expected)
    assert torch.equal(res.mlm_loss, torch.zeros_like(res.mlm_loss))
