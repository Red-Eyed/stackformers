# stackformers

Typed, composable, SOLID transformer library for PyTorch.

Behavior comes from **injected dependencies**, not constructor flags. Every architectural choice — positional encoding, normalization, feedforward variant — is a first-class object you pass in. This makes each piece independently testable, swappable without touching surrounding code, and free of `if self.x is not None` branches inside `forward()`.

---

## Design principles

| Principle | How it appears here |
|-----------|---------------------|
| Dependency injection over flags | `SelfAttention(config, pos_encoding)` — swap positional encoding without touching anything else |
| Sealed unions over optional fields | `SequenceInfo = PaddedSequence \| PackedSequence` — no god-object with nullable fields |
| Null objects over None checks | `NoPosEncoding` implements the protocol and passes q/k unchanged — `forward()` never branches on `None` |
| Static vs runtime dispatch | Attention type (global vs windowed) is fixed at construction via `window_size`; backend (padded vs packed) dispatches at runtime via `match` on `SequenceInput` |
| One concept per file | Each file can be described in one sentence and tested in isolation |
| Protocols over ABCs | Structural subtyping via `Protocol` — bring your own implementation |

---

## Quick start

### Preset — zero boilerplate

```python
import torch
from stackformers import (
    TransformerEncoder,
    plain_encoder_config,
    make_padded_input,
)

cfg   = plain_encoder_config(dim=512, heads=8, num_layers=6)
model = TransformerEncoder(cfg)

x    = torch.randn(2, 128, 512)
mask = torch.ones(2, 128, dtype=torch.bool)
out  = model(make_padded_input(x, mask))   # (2, 128, 512)
```

Same model, packed (training) input:

```python
from stackformers import make_packed_input

cu  = torch.tensor([0, 64, 128], dtype=torch.int32)
out = model(make_packed_input(x_flat, cu, max_seqlen=64))  # (128, 512)
```

GPT-style causal backbone:

```python
cfg = plain_encoder_config(dim=768, heads=12, num_layers=12, causal=True)
```

Sliding-window local attention (O(n · w)):

```python
cfg = windowed_encoder_config(dim=512, heads=8, num_layers=6, window_size=128)
```

Encoder–decoder:

```python
from stackformers import TransformerDecoder, plain_decoder_config

cfg   = plain_decoder_config(dim=512, heads=8, num_layers=6)
model = TransformerDecoder(cfg)
out   = model(make_padded_input(x, mask), make_padded_input(context, ctx_mask))
```

### Full config — explicit control

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
```

Windowed self-attention in the full config:

```python
SelfAttentionConfig(dim=512, heads=8, dim_head=64, window_size=128)
```

### JSON config round-trip

All union config fields carry a `kind` discriminator, so configs serialise and deserialise cleanly:

```python
cfg  = TransformerEncoderConfig(...)
data = cfg.model_dump()
cfg2 = TransformerEncoderConfig.model_validate(data)
```

### Custom wiring

```python
from stackformers import (
    SelfAttentionConfig, SwiGLUConfig, RMSNormConfig, RoPE1DConfig,
    SelfAttention, SwiGLU, TransformerLayer, Encoder, RMSNorm,
    RotaryEmbedding1D, make_padded_input,
)

attn_cfg = SelfAttentionConfig(dim=512, heads=8, dim_head=64)
ff_cfg   = SwiGLUConfig(dim=512)
norm_cfg = RMSNormConfig(dim=512)
pos      = RotaryEmbedding1D(RoPE1DConfig(dim_head=64))

layers = [
    TransformerLayer(
        self_attn=SelfAttention(config=attn_cfg, pos_encoding=pos),
        ff=SwiGLU(ff_cfg),
        norm_attn=RMSNorm(norm_cfg),
        norm_ff=RMSNorm(norm_cfg),
    )
    for _ in range(6)
]
encoder = Encoder(layers=layers, final_norm=RMSNorm(norm_cfg))
```

---

## Glossary

### Dimension symbols

All tensor arguments are annotated with [jaxtyping](https://github.com/patrick-kidger/jaxtyping) shape strings using these symbols consistently across the entire codebase:

| Symbol | Meaning |
|--------|---------|
| `b` | batch size |
| `n` | query sequence length |
| `s` | key / value sequence length (`s = n` for self-attention) |
| `d` | model dimension |
| `h` | number of query heads |
| `dh` | dimension per head |
| `w` | window size (local attention) |
| `nt` | total tokens in a packed sequence (`nt = Σ lengths`) |

---

## Module map

```
stackformers/
├── sequence.py              PaddedSequence, PackedSequence, PaddedInput, PackedInput
├── config.py                LayerConfig, EncoderConfig, DecoderConfig
├── layers.py                TransformerLayer (pre-norm residual)
├── encoder.py               Encoder
├── decoder.py               DecoderLayer, Decoder
├── cross_attender.py        CrossAttenderLayer, CrossAttenderStack
├── attention/
│   ├── config.py            SelfAttentionConfig, CrossAttentionConfig
│   ├── protocols.py         SelfAttn, CrossAttn
│   ├── self_attn.py         SelfAttention  (global or windowed; padded or packed)
│   └── cross_attn.py        CrossAttention (global; padded or packed)
├── feedforward/
│   ├── config.py            SwiGLUConfig, GEGLUConfig, FeedForwardConfig
│   ├── protocols.py         FeedForward
│   ├── swiglu.py            SwiGLU
│   └── factory.py           build_ff
├── norm/
│   ├── config.py            RMSNormConfig, LayerNormConfig, NormConfig
│   ├── protocols.py         Norm
│   ├── rms.py               RMSNorm
│   └── factory.py           build_norm
├── positional/
│   ├── config.py            RoPE1DConfig, RoPE2DConfig, NoPosEncodingConfig, PosEncodingConfig
│   ├── protocols.py         PosEncoding, PackedPosEncoding
│   ├── none.py              NoPosEncoding (null object)
│   ├── rope1d.py            RotaryEmbedding1D
│   ├── rope2d.py            RotaryEmbedding2D
│   └── factory.py           build_pos_encoding
└── presets/
    ├── encoder.py           TransformerEncoder, TransformerEncoderConfig,
    │                        plain_encoder_config, windowed_encoder_config
    ├── decoder.py           TransformerDecoder, TransformerDecoderConfig, plain_decoder_config
    └── cross_attender.py    CrossAttender, CrossAttenderConfig, plain_cross_attender_config
```

---

## Sequence types

```python
from stackformers import make_padded_input, make_packed_input

# Padded batch — mask is True for valid tokens
inp = make_padded_input(x, mask)                         # PaddedInput

# Packed batch (variable-length, FlashAttention convention)
inp = make_packed_input(x, cu_seqlens, max_seqlen=512)   # PackedInput
```

`SequenceInfo = PaddedSequence | PackedSequence` is a sealed union. New sequence types are new classes — existing variants are never modified.

All attention modules and presets accept either input type. Use `PaddedInput` for inference (easy masking), `PackedInput` for training (no padding waste). The model weights are shared.

---

## Attention

`SelfAttention` covers four combinations via two decisions made at different times:

| Decision | When | Controlled by |
|----------|------|--------------|
| Global vs windowed | Construction | `SelfAttentionConfig.window_size` (`None` = global) |
| Padded vs packed backend | Runtime | `SequenceInput` type passed to `forward` |

On CUDA with fp16/bf16, the packed path uses `torch.nn.attention.varlen.varlen_attn`. On CPU or fp32 it falls back to a per-sequence SDPA loop — correct on all hardware, fast where it matters.

GQA / MQA: set `kv_heads < heads` in either config.

---

## Installation

```bash
# Runtime
uv add stackformers

# Development (clone + editable)
git clone <repo>
cd stackformers
uv sync --group dev
```

---

## Development commands

```
just sync        install all dependencies
just fmt         ruff format
just lint        ruff check
just types       pyrefly check
just test        pytest
just check       fmt-check + lint + types + test  (CI gate)
```

---

## Roadmap

| Area | Status |
|------|--------|
| Padded + packed self-attention (global and windowed) | ✅ done |
| Padded + packed cross-attention | ✅ done |
| GQA / MQA | ✅ done |
| RoPE-1D, RoPE-2D, learned positional encoding | ✅ done |
| RMSNorm, LayerNorm | ✅ done |
| SwiGLU, GEGLU feedforward | ✅ done |
| Encoder, Decoder, CrossAttender building blocks | ✅ done |
| Preset configs with factory functions | ✅ done |
| FlexAttention | 🔲 planned |
| MLA (Multi-Latent Attention) | 🔲 planned |
| Sparse / mixture-of-experts FFN | 🔲 planned |
| KV-cache support | 🔲 planned |

---

## License

See [LICENSE](LICENSE).
