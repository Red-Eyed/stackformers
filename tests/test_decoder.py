"""Tests for decoder stacks and normalization topologies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from typing_extensions import override

from stackformers.attention.config import CrossAttentionConfig, SelfAttentionConfig
from stackformers.decoder import (
    Decoder,
    DecoderLayer,
    DecoderLayerBase,
    PostNormDecoderLayer,
    ReorderedNormDecoderLayer,
    SandwichNormDecoderLayer,
)
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.norm.config import NormPlacement, RMSNormConfig
from stackformers.positional.config import NoPosEncodingConfig, RoPE1DConfig
from stackformers.presets.decoder import TransformerDecoder, TransformerDecoderConfig
from stackformers.sequence import PaddedInput, make_padded_input
from tests.export_utils import ONNX_OPSET_CASES, ExportShapeMode, export_and_run
from tests.norm_topology_helpers import (
    AffineNorm,
    ScaleCrossAttention,
    ScaleFeedForward,
    ScaleSelfAttention,
    expected_cross_attention,
    expected_norm,
    make_topology_input,
)

if TYPE_CHECKING:
    from stackformers.attention.masking import SelfAttentionMask

B, N, S, D, H, DH = 2, 8, 12, 64, 4, 16


def _decoder_layer(norm_placement: NormPlacement) -> DecoderLayerBase:
    """Build a deterministic decoder layer for an exact topology check."""
    self_attn = ScaleSelfAttention(2.0)
    cross_attn = ScaleCrossAttention(3.0)
    ff = ScaleFeedForward(4.0)
    match norm_placement:
        case "pre":
            return DecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "post":
            return PostNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "sandwich":
            return SandwichNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )
        case "reordered":
            return ReorderedNormDecoderLayer(
                self_attn,
                cross_attn,
                ff,
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
                AffineNorm(5.0, 1.0),
            )


def _expected_decoder_output(
    norm_placement: NormPlacement,
    x: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the selected decoder equation independently of its implementation."""
    match norm_placement:
        case "pre":
            after_self = x + 2.0 * expected_norm(x)
            after_cross = after_self + expected_cross_attention(expected_norm(after_self), context)
            return after_cross + 4.0 * expected_norm(after_cross)
        case "post":
            after_self = expected_norm(x + 2.0 * x)
            after_cross = expected_norm(after_self + expected_cross_attention(after_self, context))
            return expected_norm(after_cross + 4.0 * after_cross)
        case "sandwich":
            after_self = x + expected_norm(2.0 * expected_norm(x))
            after_cross = after_self + expected_norm(
                expected_cross_attention(expected_norm(after_self), context)
            )
            return after_cross + expected_norm(4.0 * expected_norm(after_cross))
        case "reordered":
            after_self = x + expected_norm(2.0 * x)
            after_cross = after_self + expected_norm(expected_cross_attention(after_self, context))
            return after_cross + expected_norm(4.0 * after_cross)


def _expected_context_free_output(norm_placement: NormPlacement, x: torch.Tensor) -> torch.Tensor:
    """Evaluate only the self-attention and feed-forward residual branches."""
    match norm_placement:
        case "pre":
            x = x + 2.0 * expected_norm(x)
            return x + 4.0 * expected_norm(x)
        case "post":
            x = expected_norm(x + 2.0 * x)
            return expected_norm(x + 4.0 * x)
        case "sandwich":
            x = x + expected_norm(2.0 * expected_norm(x))
            return x + expected_norm(4.0 * expected_norm(x))
        case "reordered":
            x = x + expected_norm(2.0 * x)
            return x + expected_norm(4.0 * x)


@pytest.fixture
def config() -> TransformerDecoderConfig:
    return TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )


@pytest.fixture
def decoder(
    config: TransformerDecoderConfig,
    device_dtype: tuple[torch.device, torch.dtype],
) -> TransformerDecoder:
    device, dtype = device_dtype
    return TransformerDecoder(config).to(device=device, dtype=dtype)


@pytest.fixture
def x_context_inp(
    device_dtype: tuple[torch.device, torch.dtype],
) -> tuple[PaddedInput, PaddedInput]:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    return x_inp, ctx_inp


def test_decoder_output_shape(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    x_inp, ctx_inp = x_context_inp
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_ctx_padding(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x_inp, _ = x_context_inp
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)
    mask[1, 8:] = False
    ctx_inp = make_padded_input(context, mask)
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_tgt_padding(
    decoder: TransformerDecoder,
    device_dtype: tuple[torch.device, torch.dtype],
) -> None:
    device, dtype = device_dtype
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, 6:] = False
    x_inp = make_padded_input(x, mask)
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert decoder(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_causal_self_attn_shape() -> None:
    """Decoder with causal=True in self_attn config runs without error."""
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=1,
    )
    model = TransformerDecoder(cfg)
    x = torch.randn(B, N, D)
    context = torch.randn(B, S, D)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_with_rope(device_dtype: tuple[torch.device, torch.dtype]) -> None:
    device, dtype = device_dtype
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=RoPE1DConfig(dim_head=DH),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device, dtype=dtype)
    x = torch.randn(B, N, D, device=device, dtype=dtype)
    context = torch.randn(B, S, D, device=device, dtype=dtype)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    assert model(x_inp, ctx_inp).shape == (B, N, D)


def test_decoder_gradients(device: torch.device) -> None:
    cfg = TransformerDecoderConfig(
        self_attn=SelfAttentionConfig(dim=D, heads=H, dim_head=DH, causal=True),
        cross_attn=CrossAttentionConfig(dim=D, heads=H, dim_head=DH),
        ff=SwiGLUConfig(dim=D),
        norm=RMSNormConfig(dim=D),
        pos_encoding=NoPosEncodingConfig(),
        num_layers=2,
    )
    model = TransformerDecoder(cfg).to(device=device)
    x = torch.randn(B, N, D, device=device, requires_grad=True)
    context = torch.randn(B, S, D, device=device, requires_grad=True)
    x_inp = make_padded_input(x, torch.ones(B, N, dtype=torch.bool, device=device))
    ctx_inp = make_padded_input(context, torch.ones(B, S, dtype=torch.bool, device=device))
    model(x_inp, ctx_inp).sum().backward()
    assert x.grad is not None
    assert context.grad is not None


def test_decoder_config_accessor(config: TransformerDecoderConfig) -> None:
    assert TransformerDecoder(config).config is config


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_norm_placement_equation(norm_placement: NormPlacement) -> None:
    """Each decoder class implements its exact three-branch normalization equation."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _decoder_layer(norm_placement)

    expected = _expected_decoder_output(norm_placement, x_input.x, ctx_input.x)

    assert torch.equal(layer(x_input, ctx_input).x, expected)


def test_sandwich_decoder_uses_six_independent_norms() -> None:
    """No affine scale is tied across sandwich branch boundaries."""
    layer = _decoder_layer("sandwich")

    assert isinstance(layer, SandwichNormDecoderLayer)
    norms = [
        layer.norm_self_pre,
        layer.norm_self_post,
        layer.norm_cross_pre,
        layer.norm_cross_post,
        layer.norm_ff_pre,
        layer.norm_ff_post,
    ]
    assert len({id(norm) for norm in norms}) == 6


def test_decoder_layer_retains_legacy_state_dict_keys() -> None:
    """The pre-norm decoder constructor and checkpoint names remain unchanged."""
    layer = _decoder_layer("pre")

    assert list(layer.state_dict()) == [
        "self_attn.scale",
        "cross_attn.query_scale",
        "cross_attn.context_scale",
        "ff.scale",
        "norm_self.scale",
        "norm_self.offset",
        "norm_cross.scale",
        "norm_cross.offset",
        "norm_ff.scale",
        "norm_ff.offset",
    ]


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_norm_placement_gradients(norm_placement: NormPlacement) -> None:
    """Every decoder topology propagates gradients through target and context inputs."""
    x_input = make_topology_input(requires_grad=True)
    ctx_input = make_topology_input(12, requires_grad=True)
    layer = _decoder_layer(norm_placement)

    layer(x_input, ctx_input).x.sum().backward()

    assert x_input.x.grad is not None
    assert ctx_input.x.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", ONNX_OPSET_CASES)
@pytest.mark.parametrize("with_context", [False, True])
def test_decoder_norm_placement_is_export_compatible(
    norm_placement: NormPlacement,
    shape_mode: ExportShapeMode,
    opset_version: int,
    with_context: bool,
) -> None:
    """Every decoder topology supports static and dynamic export."""
    x_input = make_topology_input()
    ctx_input = make_topology_input(12)
    layer = _decoder_layer(norm_placement)
    resized_x_input = make_padded_input(
        torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )
    resized_ctx_input = make_padded_input(
        torch.arange(12, 28, dtype=torch.float32).reshape(1, 4, 4),
        torch.ones(1, 4, dtype=torch.bool),
    )
    query_tokens = torch.export.Dim("query_tokens", min=1, max=4)
    context_tokens = torch.export.Dim("context_tokens", min=1, max=5)
    shapes = torch.export.ShapesCollection()
    for tensor in x_input:
        shapes[tensor] = {1: query_tokens}
    if with_context:
        for tensor in ctx_input:
            shapes[tensor] = {1: context_tokens}
    args = (x_input, ctx_input) if with_context else (x_input,)
    runtime_args = (resized_x_input, resized_ctx_input) if with_context else (resized_x_input,)

    export_and_run(
        layer,
        args,
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(layer, args),
        runtime_args=runtime_args,
    )


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_without_context_skips_cross_branch(norm_placement: NormPlacement) -> None:
    """Omission skips cross norms and parameters, including their affine offsets."""
    layer = _decoder_layer(norm_placement)
    input = make_topology_input(requires_grad=True)
    expected = _expected_context_free_output(norm_placement, input.x)
    actual = layer(input)
    assert torch.equal(actual.x, expected)
    assert torch.equal(layer(input, None).x, expected)
    actual.x.sum().backward()
    assert input.x.grad is not None
    for name, parameter in layer.named_parameters():
        assert (parameter.grad is None) == ("cross" in name)


def test_decoder_preset_without_context(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    """The public preset defaults to context-free decoding."""
    input, _ = x_context_inp
    assert decoder(input).shape == (B, N, D)
    assert torch.equal(decoder(input), decoder(input, None))


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_context_free_decoder_compiles_and_exports(norm_placement: NormPlacement) -> None:
    """Context absence specializes a full graph without tensor-dependent control flow."""
    decoder = Decoder([_decoder_layer(norm_placement)], torch.nn.Identity())
    input = make_topology_input()
    expected = decoder(input)
    compiled = torch.compile(decoder, backend="eager", fullgraph=True)
    assert torch.equal(compiled(input), expected)
    exported = torch.export.export(decoder, (input,)).module()
    assert torch.equal(exported(input), expected)


@pytest.fixture
def visibility_decoder(
    config: TransformerDecoderConfig,
    norm_placement: NormPlacement,
    device_dtype: tuple[torch.device, torch.dtype],
) -> TransformerDecoder:
    """Build every topology with the same runtime visibility contract."""
    device, dtype = device_dtype
    return (
        TransformerDecoder(config.model_copy(update={"norm_placement": norm_placement}))
        .to(device=device, dtype=dtype)
        .eval()
    )


@pytest.mark.parametrize("norm_placement", ["pre", "post", "sandwich", "reordered"])
def test_decoder_runtime_masks(
    visibility_decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    """Mode/tensor equivalence and excluded-context invariance hold across the full stack."""
    from tests.conftest import atol

    input, context = x_context_inp
    causal = torch.ones(N, N, device=input.x.device, dtype=torch.bool).tril()
    full = torch.ones_like(causal)
    cross = torch.zeros(N, S, device=input.x.device, dtype=torch.bool)
    cross[:, :2] = True
    changed = context.x.clone()
    changed[:, 2:] += 100
    changed_context = context._replace(x=changed)

    modes: tuple[tuple[SelfAttentionMask, torch.Tensor], ...] = (
        ("causal", causal),
        ("bidirectional", full),
    )
    for mode, mask in modes:
        actual = visibility_decoder(input, context, self_attn_mask=mode, cross_attn_mask=cross)
        expected = visibility_decoder(
            input, changed_context, self_attn_mask=mask, cross_attn_mask=cross
        )
        torch.testing.assert_close(actual, expected, atol=atol(actual.dtype), rtol=0)
    torch.testing.assert_close(
        visibility_decoder(input), visibility_decoder(input, self_attn_mask=causal)
    )


def test_decoder_rejects_cross_mask_without_context(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    """A context mask must never be accepted and silently ignored."""
    input, _ = x_context_inp
    with pytest.raises(ValueError, match="requires context"):
        decoder(input, cross_attn_mask=torch.ones(N, S, device=input.x.device, dtype=torch.bool))


def test_decoder_tensor_masks_compile_and_export(
    decoder: TransformerDecoder,
    x_context_inp: tuple[PaddedInput, PaddedInput],
) -> None:
    """Masks remain graph inputs whose values can change after tracing."""
    input, context = x_context_inp
    self_mask = torch.ones(N, N, device=input.x.device, dtype=torch.bool).tril()
    cross_mask = torch.ones(N, S, device=input.x.device, dtype=torch.bool)
    compiled = torch.compile(decoder, backend="eager", fullgraph=True)
    exported = torch.export.export(
        decoder,
        (input, context),
        kwargs={"self_attn_mask": self_mask, "cross_attn_mask": cross_mask},
    ).module()
    for allowed in [False, True]:
        cross_mask.fill_(allowed)
        expected = decoder(input, context, self_attn_mask=self_mask, cross_attn_mask=cross_mask)
        torch.testing.assert_close(
            compiled(input, context, self_attn_mask=self_mask, cross_attn_mask=cross_mask), expected
        )
        torch.testing.assert_close(
            exported(input, context, self_attn_mask=self_mask, cross_attn_mask=cross_mask), expected
        )


@pytest.fixture
def gradient_decoder(config: TransformerDecoderConfig, device: torch.device) -> TransformerDecoder:
    """Use float32 on each available device for visibility-gradient checks."""
    return TransformerDecoder(config).to(device=device)


@pytest.fixture
def gradient_inputs(device: torch.device) -> tuple[PaddedInput, PaddedInput]:
    """Keep original leaf tensors available for checking excluded-token gradients."""
    input = make_padded_input(
        torch.randn(B, N, D, device=device, requires_grad=True),
        torch.ones(B, N, device=device, dtype=torch.bool),
    )
    context = make_padded_input(
        torch.randn(B, S, D, device=device, requires_grad=True),
        torch.ones(B, S, device=device, dtype=torch.bool),
    )
    return input, context


@pytest.mark.parametrize("packed", [False, True])
def test_decoder_masks_exclude_gradients(
    gradient_decoder: TransformerDecoder,
    gradient_inputs: tuple[PaddedInput, PaddedInput],
    packed: bool,
) -> None:
    """Neither future target tokens nor excluded image tokens receive gradients."""
    from stackformers.sequence import padded_to_packed

    input, context = gradient_inputs
    allowed = torch.zeros(N, S, dtype=torch.bool, device=input.x.device)
    allowed[:, :2] = True
    if packed:
        output = gradient_decoder(
            padded_to_packed(input),
            padded_to_packed(context),
            self_attn_mask="causal",
            cross_attn_mask=allowed,
        )
        loss = output[0].sum()
    else:
        output = gradient_decoder(input, context, self_attn_mask="causal", cross_attn_mask=allowed)
        loss = output[0, 0].sum()
    loss.backward()
    assert input.x.grad is not None
    assert context.x.grad is not None
    assert torch.isfinite(input.x.grad).all()
    assert torch.isfinite(context.x.grad).all()
    assert input.x.grad[0, 1:].eq(0).all()
    assert context.x.grad[0, 2:].eq(0).all()
    assert context.x.grad[0, :2].abs().sum() > 0
    assert input.x.grad[1].eq(0).all()
    assert context.x.grad[1].eq(0).all()


class MaskedDecoderExport(torch.nn.Module):
    """Expose keyword-only visibility as tensor inputs for shared ONNX test utilities."""

    def __init__(self, decoder: TransformerDecoder) -> None:
        """Keep the ordinary decoder's weights and per-call mask semantics."""
        super().__init__()
        self.decoder = decoder

    @override
    def forward(
        self,
        input: PaddedInput,
        context: PaddedInput,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode with explicit shared query/key masks."""
        return self.decoder(input, context, self_attn_mask=self_mask, cross_attn_mask=cross_mask)


@pytest.fixture
def masked_decoder_export(config: TransformerDecoderConfig) -> MaskedDecoderExport:
    """ONNX Runtime executes this CPU float32 decoder."""
    return MaskedDecoderExport(TransformerDecoder(config).eval())


@pytest.mark.parametrize("shape_mode", tuple(ExportShapeMode), ids=lambda mode: mode.value)
@pytest.mark.parametrize("opset_version", [18, 23])
def test_decoder_runtime_masks_onnx(
    masked_decoder_export: MaskedDecoderExport,
    shape_mode: ExportShapeMode,
    opset_version: int,
) -> None:
    """Both ONNX attention lowerings retain dynamic mask axes and fully excluded rows."""
    input = make_padded_input(torch.randn(B, N, D), torch.ones(B, N, dtype=torch.bool))
    context = make_padded_input(torch.randn(B, S, D), torch.ones(B, S, dtype=torch.bool))
    self_mask = torch.ones(N, N, dtype=torch.bool).tril()
    cross_mask = torch.ones(N, S, dtype=torch.bool)
    cross_mask[0] = False
    args = (input, context, self_mask, cross_mask)
    n, s = torch.export.Dim("n", min=1, max=N), torch.export.Dim("s", min=1, max=S)
    shapes = torch.export.ShapesCollection()
    for tensor in input:
        shapes[tensor] = {1: n}
    for tensor in context:
        shapes[tensor] = {1: s}
    shapes[self_mask] = {0: n, 1: n}
    shapes[cross_mask] = {0: n, 1: s}
    resized_input = PaddedInput(*(tensor[:, :3] for tensor in input))
    resized_context = PaddedInput(*(tensor[:, :5] for tensor in context))
    export_and_run(
        masked_decoder_export,
        args,
        shape_mode,
        opset_version,
        dynamic_shapes=shapes.dynamic_shapes(masked_decoder_export, args),
        runtime_args=(resized_input, resized_context, self_mask[:3, :3], cross_mask[:3, :5]),
    )
