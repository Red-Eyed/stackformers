# stackformers

Typed, composable transformer library for PyTorch.
Every architectural choice — positional encoding, normalization, feedforward variant — is an injected dependency, not a constructor flag.

```bash
uv add stackformers
```

Runnable examples cover variable-length layouts, causal language modeling, image patches,
continuous point coordinates, user subclassing, and cached ONNX Runtime deployment. See the
[`examples` guide](examples/README.md).

---

## Why

Most transformer libraries grow into a tangle of `if self.use_rope`, `if self.window_size is not None`, and god-config objects with thirty nullable fields. Adding a new variant means touching existing code.

stackformers takes a different approach:

- **Swap any component without touching anything else** — `SelfAttention(config, pos_encoding=RoPE)` vs `SelfAttention(config, pos_encoding=ALiBi)` — same call site, different object
- **No `None` checks in `forward()`** — `NoPosEncoding` is a real object that passes q/k unchanged; the branch never exists
- **Sealed sequence unions** — `PaddedInput | PackedInput` instead of optional `cu_seqlens` and `mask` arguments that conflict with each other
- **`torch.compile` / `torch.export` safe** — no Python control flow on tensors inside any `forward()`
- **Structural protocols** — bring your own implementation; no ABC inheritance required

---

## Quick start

### Zero boilerplate

```python
import torch
from stackformers import TransformerEncoder, plain_encoder_config, make_padded_input

model = TransformerEncoder(plain_encoder_config(dim=512, heads=8, num_layers=6))

x    = torch.randn(2, 128, 512)
mask = torch.ones(2, 128, dtype=torch.bool)
out  = model(make_padded_input(x, mask))   # (2, 128, 512)
```

Switch to packed (variable-length, no padding waste) — same weights:

```python
from stackformers import make_packed_input

cu  = torch.tensor([0, 64, 128], dtype=torch.int32)
out = model(make_packed_input(x_flat, cu, max_seqlen=64))  # (128, 512)
```

Causal LM backbone:

```python
plain_encoder_config(dim=768, heads=12, num_layers=12, causal=True)
```

Sliding-window local attention (O(n · w)):

```python
from stackformers import windowed_encoder_config
windowed_encoder_config(dim=512, heads=8, num_layers=6, window_size=128)
```

Variable-width encoder — one model and head dimension per Transformer block:

```python
from stackformers import VariableWidthTransformerEncoder, variable_width_encoder_config

config = variable_width_encoder_config(
    d_models=[512, 512, 768, 768, 1024],
    dim_heads=[64, 64, 64, 64, 128],
)
model = VariableWidthTransformerEncoder(config)
```

The preset derives each block's head count as `d_model // dim_head`. It inserts a bias-free
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
from stackformers import TransformerDecoder, plain_decoder_config

model = TransformerDecoder(plain_decoder_config(dim=512, heads=8, num_layers=6))
out   = model(make_padded_input(x, mask), make_padded_input(context, ctx_mask))
```

For autoregressive export, build cross K/V once and pass both required cache tensors to every
one-token decoder invocation:

```python
import torch

from stackformers import (
    CachedDecoderWrapper,
    DecoderCrossAttentionCacheBuilder,
)

model.eval()
cache_builder = DecoderCrossAttentionCacheBuilder(model).eval()
cached_decoder = CachedDecoderWrapper(model).eval()

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
[`examples/encoder_decoder_onnxruntime.py`](examples/encoder_decoder_onnxruntime.py).

### Explicit config

Full control with JSON round-trip via `kind` discriminators:

```python
from stackformers import (
    TransformerEncoderConfig, TransformerEncoder,
    SelfAttentionConfig, SwiGLUConfig, RMSNormConfig, RoPE1DConfig,
    make_padded_input,
)

cfg = TransformerEncoderConfig(
    attn=SelfAttentionConfig(dim=512, heads=8, dim_head=64, causal=False),
    ff=SwiGLUConfig(dim=512, mult=4.0),
    norm=RMSNormConfig(dim=512),
    pos_encoding=RoPE1DConfig(dim_head=64),
    num_layers=6,
)
model = TransformerEncoder(cfg)

# Serialise / restore
cfg2 = TransformerEncoderConfig.model_validate(cfg.model_dump())
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

### Custom wiring

Wire layers yourself when presets aren't enough:

```python
from stackformers import (
    SelfAttention, SwiGLU, TransformerLayer, Encoder, RMSNorm,
    RotaryEmbedding1D,
    SelfAttentionConfig, SwiGLUConfig, RMSNormConfig, RoPE1DConfig,
)

pos  = RotaryEmbedding1D(RoPE1DConfig(dim_head=64))
attn = SelfAttention(SelfAttentionConfig(dim=512, heads=8, dim_head=64), pos_encoding=pos)

layers = [
    TransformerLayer(
        self_attn=attn,
        ff=SwiGLU(SwiGLUConfig(dim=512)),
        norm_attn=RMSNorm(RMSNormConfig(dim=512)),
        norm_ff=RMSNorm(RMSNormConfig(dim=512)),
    )
    for _ in range(6)
]
encoder = Encoder(layers=layers, final_norm=RMSNorm(RMSNormConfig(dim=512)))
```

---

## What's included

| Area | Variants |
|------|----------|
| Self-attention | Global, sliding-window (local); padded and packed backends; GQA / MQA |
| Cross-attention | Global; padded and packed backends |
| Positional encoding | RoPE-1D, RoPE-2D, none (null object) |
| Feedforward | SwiGLU, HardSwishGLU, GEGLU, GELU, ReLU² |
| Normalization | RMSNorm, LayerNorm |
| Presets | Encoder, Decoder, CrossAttender |

On CUDA with fp16/bf16 the packed path uses `torch.nn.attention.varlen.varlen_attn`. CPU and fp32 fall back to a scatter-to-padded SDPA — correct everywhere, fast where it matters.

---

## Development

```bash
git clone <repo> && cd stackformers
uv sync --group dev

just fmt      # format
just lint     # lint
just types    # type-check
just test     # test
just check    # full CI gate
```

---

## License

See [LICENSE](LICENSE).
