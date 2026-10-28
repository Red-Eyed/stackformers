# API and configuration guide

Typed, composable transformer library for PyTorch.
Every architectural choice — positional encoding, normalization, feedforward variant — is an injected dependency, not a constructor flag.

```bash
uv add stackformers
```

Runnable examples cover variable-length layouts, causal language modeling, image patches,
continuous point coordinates, user subclassing, and cached ONNX Runtime deployment. See the
[`examples` guide](../examples/README.md).

---

## Tensor batches and conversions

`PackedTensor` and `PaddedTensor` are the public value types: each carries token values,
layout information, and absolute positions. Pass either directly to a model.

```python
from stackformers import PackedTensor, PaddedTensor

packed = PackedTensor.from_list(sequences)  # each item has shape (length, features)
padded = packed.to_padded(padding_value=0.0)
packed_again = padded.to_packed()
items = padded.to_list()
nested = packed.to_nested()  # torch.jagged, sharing the packed values buffer
restored = PaddedTensor.from_nested(nested)
```

Both constructors accept zero-length sequences but require a nonempty list with matching
feature widths, dtypes, and devices. `from_list(..., abs_positions=positions)` accepts one
coordinate tensor per item; otherwise positions start at zero within each sequence.
Layout conversions preserve custom coordinates and valid-token order. Padded output is
padded at the end of each row to the maximum document length; converting a
mask with holes compacts its valid tokens. Unused coordinates are zero.

These operations never mutate inputs and preserve autograd history, dtype, and device.
`to_list()` and `to_nested()` export values only, so reconstructing from them generates new
sequential coordinates. `to_list()` may synchronize GPU lengths; `from_list`, `from_nested`,
and padded-to-packed conversion are eager adapters. Packed-to-padded conversion supports
compile/export with dynamic output shapes. Same-layout conversions return the original
record. Packed list slices and jagged output share value storage; callers must avoid
mutating shared buffers. Nested adapters require PyTorch's jagged tensor API.

### Deprecated sequence API

`PackedInput` and `PaddedInput` remain exact aliases of the canonical types and warn with
`DeprecationWarning` when accessed. Replace their imports with `PackedTensor` and
`PaddedTensor`; direct constructor arguments and model calls remain compatible.
`PackedSequence`, `PaddedSequence`, `SequenceInfo`, `to_seq_info`, `make_packed`,
`make_padded`, `make_packed_input`, `make_padded_input`, `packed_batch_size`,
`padded_to_key_padding_mask`, `packed_to_padded`, and `padded_to_packed` remain available
as deprecated compatibility APIs. Metadata records and helper functions use static
PEP 702 deprecation markers without runtime warning wrappers, keeping old graph paths
traceable. Use tensor constructors, conversion methods, `materialize_padding_mask`, and
`tensor.cu_seqlens.shape[0] - 1` instead. Existing decoder cache outputs retain their
mask-only metadata structure; cached calls also accept `PaddedTensor` directly.
No removal version is assigned.

## Composition

Attention receives positional encoding as a collaborator. To change the encoding, supply a
new object satisfying its protocol. `NoPosEncoding` implements the same interface for paths
that do not need positional information. Config factories build these collaborators from
explicit discriminated configuration records.

`PaddedTensor | PackedTensor` represents the input layout, and structural protocols let custom
components integrate without inheriting a library base class. Compile/export support is
path-specific; see the [compatibility guide](compatibility.md).

## Quick start

Import the public API with `import stackformers as sf`. Configs, models, protocols,
base classes, factories, sequence helpers, and projection conversions are available
directly through `sf`.

### Zero boilerplate

```python
import torch
import stackformers as sf

model = sf.TransformerEncoder(sf.plain_encoder_config(dim=512, heads=8, num_layers=6))

sequences = [torch.randn(128, 512), torch.randn(64, 512)]
padded = sf.PaddedTensor.from_list(sequences)
out = model(padded)  # (2, 128, 512)
```

Switch to packed (variable-length, no padding waste) — same weights:

```python
packed = padded.to_packed()
out = model(packed)  # (192, 512)
```

Causal LM backbone:

```python
sf.plain_encoder_config(dim=768, heads=12, num_layers=12, causal=True)
```

Sliding-window local attention (O(n · w)):

```python
sf.windowed_encoder_config(dim=512, heads=8, num_layers=6, window_size=128)
```

Variable-width encoder — one model and head dimension per Transformer block:

```python
config = sf.variable_width_encoder_config(
    d_models=[512, 512, 768, 768, 1024],
    dim_heads=[64, 64, 64, 64, 128],
)
model = sf.VariableWidthTransformerEncoder(config)
```

Omitting `heads` derives each block's head count as `d_model // dim_head`. Set `heads=12` or
`heads=[12, 12, 16, 16, 16]` to choose the internal attention width independently of `d_models`.
`dim_heads`, `heads`, `causal`, `ff_mult`, `dropout`, and `norm_placement` accept either a shared
value or a list with one entry per block. The preset inserts a bias-free
learned projection before a block when its residual width differs from the preceding block;
equal-width neighbors use an identity. The convenience factory expands each width pair into a
`VariableWidthEncoderLayerConfig` whose attention, feed-forward, norm, positional encoding, and
attention bias are all explicit. Construct these layer configs directly to replace any component;
the model does not rebuild or override them. The input width must match the first layer's
`attn.dim`, and the output uses the last layer's width.

Reference: [Wu et al., “Variable-Width Transformers” (2026)](https://arxiv.org/abs/2606.18246).
The paper uses parameter-free residual resizing; this preset instead makes each width transition
a learned linear projection, so it is a related experimental variant rather than an exact replica.

Encoder–decoder:

```python
model = sf.TransformerDecoder(sf.plain_decoder_config(dim=512, heads=8, num_layers=6))
target_input = sf.PaddedTensor.from_list([torch.randn(8, 512), torch.randn(5, 512)])
context_input = sf.PaddedTensor.from_list([torch.randn(12, 512), torch.randn(9, 512)])
out = model(target_input, context_input)
```

For autoregressive export, build cross K/V once and pass both required cache tensors to every
one-token decoder invocation:

```python
import torch

import stackformers as sf

model.eval()
cache_builder = sf.DecoderCrossAttentionCacheBuilder(model).eval()
cached_decoder = sf.CachedDecoderWrapper(model).eval()

cross_cache = cache_builder(context_input)
self_kv_cache = torch.empty(num_layers, 2, batch, kv_heads, 0, dim_head)
step_i = torch.tensor([0], dtype=torch.int64)
cache_onnx = torch.onnx.export(cache_builder, (context_input,), dynamo=True)
decoder_onnx = torch.onnx.export(
    cached_decoder,
    (target_token, cross_cache, self_kv_cache, step_i),
    dynamo=True,
)
```

The cache builder runs once per encoder context. Each decoder call consumes exactly one target
token, the immutable cross cache, the growing self cache, and `step_i`; it returns the output token
and a self cache extended by one position. K/V tensors retain `kv_heads` rather than expanded query
heads. Cache construction positions cross K once, while cached decoding positions each new self K
and both self/cross Q, so all built-in positional encodings remain available. Batch, source, and
past-target axes are dynamic during export; model width, layer count, head geometry, and the
one-token axis remain static because weights constrain them.

See the complete eager, ONNX export, and raw ONNX Runtime demonstration in
[`examples/encoder_decoder_onnxruntime.py`](../examples/encoder_decoder_onnxruntime.py).

### Explicit config

Full control with JSON round-trip via `kind` discriminators:

```python
import stackformers as sf

cfg = sf.TransformerEncoderConfig(
    attn=sf.SelfAttentionConfig(dim=512, heads=8, dim_head=64, causal=False),
    ff=sf.SwiGLUConfig(dim=512, mult=4.0),
    norm=sf.RMSNormConfig(dim=512),
    pos_encoding=sf.RoPE1DConfig(dim_head=64),
    num_layers=6,
)
model = sf.TransformerEncoder(cfg)

# Serialise / restore
cfg2 = sf.TransformerEncoderConfig.model_validate(cfg.model_dump())
```

`norm_placement` is available on `TransformerEncoderConfig`, `TransformerDecoderConfig`, and
`CrossAttenderConfig`, as well as their plain-config helpers. It accepts four layouts and defaults
to `"pre"`, so existing constructors, serialized configs, checkpoints, outputs, and gradients retain
the previous behavior when the field is omitted. For a residual branch `F` with norm `N`:

| Value | Branch equation | Reference |
| --- | --- | --- |
| `"pre"` | `x + F(N(x))` | [Xiong et al., 2020](https://proceedings.mlr.press/v119/xiong20b.html) |
| `"post"` | `N(x + F(x))` | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| `"sandwich"` | `x + N_post(F(N_pre(x)))` | [Ding et al., 2021](https://arxiv.org/abs/2105.13290) |
| `"reordered"` | `x + N(F(x))` | [Liu et al., 2022](https://arxiv.org/abs/2111.09883); [OLMo Team et al., 2025](https://arxiv.org/abs/2501.00656) |

Sandwich placement creates independent pre- and post-branch norms. Reordered placement follows
the OLMo 2 residual layout; enable QK-Norm separately in the attention config when reproducing the
broader OLMo 2 stabilization recipe. Preset construction maps each value to a focused encoder,
decoder, or cross-attender layer class. Decoder and cross-attender placement applies to the
target/query residual stream; the context sequence is not normalized or mutated by these layers.
Ordinary and cached decoder layers call the same normalization/residual equation functions.
Attention execution is supplied through small call adapters, while norm and projection modules
remain registered under their existing checkpoint names. Internal context variants encode either
omission or a complete context value; cached contexts keep K/V and their validity mask together.

Cached deployment adapters support exact built-in attention, decoder-stack, and decoder-layer
classes. Unsupported subclasses raise `TypeError` because the adapters cannot infer overridden
forward behavior. Custom collaborators continue to work in ordinary decoding. Direct cached-layer
calls require both cross cache and context, or neither; a half-present pair raises `ValueError`.

### Custom wiring

Wire layers yourself when presets aren't enough:

```python
import stackformers as sf

pos  = sf.RotaryEmbedding1D(sf.RoPE1DConfig(dim_head=64))
attention_config = sf.SelfAttentionConfig(dim=512, heads=8, dim_head=64)

layers = [
    sf.TransformerLayer(
        self_attn=sf.SelfAttention(attention_config, pos_encoding=pos),
        ff=sf.SwiGLU(sf.SwiGLUConfig(dim=512)),
        norm_attn=sf.build_norm(sf.RMSNormConfig(dim=512)),
        norm_ff=sf.build_norm(sf.RMSNormConfig(dim=512)),
    )
    for _ in range(6)
]
encoder = sf.Encoder(layers=layers, final_norm=sf.build_norm(sf.RMSNormConfig(dim=512)))
```

---

## Convert projection layouts

Load an ordinary checkpoint, then recursively pack supported self-attention, cross-attention,
SwiGLU, GEGLU, and HardSwishGLU layers:

```python
import stackformers as sf

model.load_state_dict(checkpoint)
model = sf.to_packed(model)
# Run or save the packed model.
model = sf.to_unpacked(model)
```

These helpers replace registered children in place and return the root, which may itself be
replaced. Repeated calls keep layers already in the requested layout. Shared layer references
and retained child modules stay shared. Built-in preset configs follow the new layout; save
`model.config.model_dump_json()` alongside its state dict to reconstruct that layout later.
Caller-owned config objects are left unchanged. Custom subclasses keep their own behavior
and metadata; supported children are still visited.

Weights round-trip exactly; outputs match within dtype tolerance because matrix multiplication
rounding can change. Convert before constructing an optimizer: replaced projection parameters
do not migrate optimizer state, existing gradients, hooks, or cross-layer parameter aliases.
Groups with mixed `requires_grad` fail before mutation. Cache wrappers support both ordinary and
packed attention with the same calls and cache tensor layouts. Pack the model before constructing
cache wrappers, or convert a containing module that registers the model and its existing wrappers
together so their attention references remain shared. Packing a wrapper alone converts only its
registered children; references outside that tree still point to the original attention modules.
Projection packing is independent of the `PackedTensor` sequence layout.

For deployments with no padding, `PaddedTensor(mask=AllValid(), ...)` selects mask-free global
attention. `AllValidCachedDecoderWrapper.empty_cache(dtype=...)` derives per-layer self-cache
geometry automatically.
`AllValidCachedDecoderWrapper` provides functional one-token decoding at ONNX opsets 23/24;
`PreallocatedCachedDecoderWrapper` adds a separate fixed-capacity opset-24 strategy with
explicit ownership and ORT buffer-binding requirements. See the
[inference contracts and examples](inference.md) before selecting either path.

## What's included

| Area | Variants |
|------|----------|
| Self-attention | Global, sliding-window (local); padded and packed backends; GQA / MQA |
| Cross-attention | Global; padded and packed backends |
| Positional encoding | RoPE-1D, RoPE-2D, none (null object) |
| Feedforward | SwiGLU, HardSwishGLU, GEGLU, GELU, ReLU² |
| Normalization | RMSNorm, LayerNorm |
| Presets | Encoder, Decoder, CrossAttender |

On CUDA with fp16/bf16 the packed path uses `torch.nn.attention.varlen.varlen_attn`. CPU and fp32 fall back to a scatter-to-padded SDPA — performance depends on hardware, dtype, and input lengths.

---

## Development

```bash
git clone https://github.com/Red-Eyed/stackformers.git
cd stackformers
uv sync --group dev

just fmt      # format
just lint     # lint
just types    # type-check
just test     # test
just check    # full CI gate
```

The quality gate checks the library, tests, and examples against Python 3.11. Ruff enforces
annotations, common bug patterns, typing imports, and qualified, current suppressions.
Pyrefly uses the strict preset with explicit/implicit `Any`, untyped returns, missing
annotations, and stale suppressions treated as errors. Tool minimums are recorded in both
the development dependencies and checker configs.

Module classes expose `__call__ = forward` only under `TYPE_CHECKING`. This gives callers the
declared input/output contract while leaving PyTorch's hooks and runtime dispatch intact.
Local `Tensor` annotations at built-in PyTorch projection boundaries rely on those modules'
tensor-return contracts; they do not validate tensor shapes. Experimental kernel adapters
capture the vendor's declared return type and validate the actual response before exposing
`Result[Tensor, VarlenFallback]`. Export shape trees use opaque `object` leaves because
PyTorch validates their module-dependent structure.

Core admission checks return `Result[T, E]` from `stackformers.result`, the closed union
`Ok[T] | Err[E]`. Both variants are generic NamedTuples with a value or error and a literal
kind tag. Match `Ok(value=value)` and `Err(error=error)` before accessing their payloads;
keyword class patterns work with strict PyTorch export and full-graph compilation. No
external Result package or unchecked unwrap method is required.

Recoverable core failures remain error data. Vendor adapters catch expected third-party
exceptions and preserve their original causes; cache admission validates before construction
without raising and catching its own expected failures. Existing public adapters retain
tensor/configuration returns and raise when rejected inputs prevent that operation from
proceeding. Pydantic still presents configuration failures as `ValidationError`; the
experimental backend retains tensor-or-`None` returns and fallback warnings. Unexpected
programming and PyTorch failures can propagate. See [Result records](../stackformers/result/README.md).

---

## License

See [LICENSE](../LICENSE).
