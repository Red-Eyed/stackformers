# stackformers

Typed, composable, SOLID transformer library for PyTorch.

Behavior comes from **injected dependencies**, not constructor flags. Every architectural choice — positional encoding, attention kernel, bias strategy — is a first-class object you pass in. This makes each piece independently testable, swappable without rewriting surrounding code, and free of `if self.x is not None` branches inside `forward()`.

---

## Design principles

| Principle | How it appears here |
|-----------|---------------------|
| Dependency injection over flags | `SelfAttention(config, pos_encoding, bias_builder, kernel)` — swap any piece without touching the rest |
| Sealed unions over optional fields | `SequenceInfo = PaddedSequence \| PackedSequence` — no god-object with nullable fields |
| Null objects over None checks | `NoPosEncoding`, `NoBiasBuilder` implement protocols and do nothing — `forward()` never branches on `None` |
| One concept per file | Each file can be described in one sentence and tested in isolation |
| Protocols over ABCs | Structural subtyping via `@runtime_checkable Protocol` — bring your own implementation |

---

## Quick start

### Preset — zero boilerplate

```python
import torch
from stackformers import (
    TransformerEncoder, TransformerEncoderConfig,
    AttentionConfig, FeedForwardConfig, RMSNormConfig, RoPE1DConfig,
    make_padded_input,
)

cfg = TransformerEncoderConfig(
    attn=AttentionConfig(dim=512, heads=8, dim_head=64),
    ff=FeedForwardConfig(dim=512),
    norm=RMSNormConfig(dim=512),
    pos_encoding=RoPE1DConfig(dim_head=64),
    num_layers=6,
)
model = TransformerEncoder(cfg)

x    = torch.randn(2, 128, 512)
mask = torch.ones(2, 128, dtype=torch.bool)
out  = model(make_padded_input(x, mask))   # (2, 128, 512)
```

GPT-style causal backbone:

```python
cfg = TransformerEncoderConfig(
    attn=AttentionConfig(dim=768, heads=12, dim_head=64, causal=True),
    ff=FeedForwardConfig(dim=768),
    norm=RMSNormConfig(dim=768),
    pos_encoding=RoPE1DConfig(dim_head=64),
    num_layers=12,
)
```

Sliding-window local attention (swap one config field):

```python
from stackformers import WindowedSDPAKernelConfig

cfg = TransformerEncoderConfig(
    attn=AttentionConfig(dim=512, heads=8, dim_head=64),
    ff=FeedForwardConfig(dim=512),
    norm=RMSNormConfig(dim=512),
    pos_encoding=RoPE1DConfig(dim_head=64),
    kernel=WindowedSDPAKernelConfig(window_size=128),
    num_layers=6,
)
```

Encoder–decoder:

```python
from stackformers import TransformerDecoder, TransformerDecoderConfig

cfg = TransformerDecoderConfig(
    self_attn=AttentionConfig(dim=512, heads=8, dim_head=64),
    cross_attn=AttentionConfig(dim=512, heads=8, dim_head=64),
    ff=FeedForwardConfig(dim=512),
    norm=RMSNormConfig(dim=512),
    pos_encoding=RoPE1DConfig(dim_head=64),
    num_layers=6,
)
model = TransformerDecoder(cfg)
out = model(make_padded_input(x, mask), make_padded_input(context, ctx_mask))
```

### JSON config round-trip

All union config fields carry a `kind` discriminator field, so configs serialise and deserialise cleanly:

```python
import json
from stackformers import TransformerEncoderConfig

cfg  = TransformerEncoderConfig(...)
data = cfg.model_dump()          # → dict with kind tags
cfg2 = TransformerEncoderConfig.model_validate(data)  # ← reconstructed
```

### Custom wiring — swap every piece

```python
from stackformers import (
    AttentionConfig, FeedForwardConfig, RMSNormConfig, RoPE1DConfig,
    SelfAttention, SwiGLU, TransformerLayer, Encoder, RMSNorm,
    RotaryEmbedding1D, ALiBiBuilder, SDPAKernel,
    make_padded_input,
)

attn_cfg = AttentionConfig(dim=512, heads=8, dim_head=64)
ff_cfg   = FeedForwardConfig(dim=512)

layers = [
    TransformerLayer(
        self_attn=SelfAttention(
            config=attn_cfg,
            pos_encoding=RotaryEmbedding1D(RoPE1DConfig(dim_head=64)),
            bias_builder=ALiBiBuilder(heads=8),
            kernel=SDPAKernel(),
        ),
        ff=SwiGLU(ff_cfg),
        norm_attn=RMSNorm(RMSNormConfig(dim=512)),
        norm_ff=RMSNorm(RMSNormConfig(dim=512)),
    )
    for _ in range(6)
]
encoder = Encoder(layers=layers, final_norm=RMSNorm(RMSNormConfig(dim=512)))
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
| `dh` | dimension per head (`d = h × dh`) |
| `w` | window size (local attention) |
| `nt` | total tokens in a packed sequence (`nt = Σ lengths`) |

### Common tensor shapes

| Shape | Where it appears | Meaning |
|-------|-----------------|---------|
| `b n d` | module inputs / outputs | token embeddings — one vector per position |
| `b h n dh` | inside attention | query heads, padded batch |
| `b h s dh` | inside attention | key / value heads, padded batch |
| `b 1 n s` | attention mask | additive bias broadcast over heads; `0` = keep, `-inf` = mask |
| `h n s` | attention bias | per-head additive bias (e.g. ALiBi) |
| `nt h dh` | packed attention | flat token × head tensor; no batch dim |
| `bp1` | packed sequence | cumulative sequence lengths, length `batch + 1` |

---

## Module map

```
stackformers/
├── sequence.py              PaddedSequence, PackedSequence, PaddedInput, PackedInput, SequenceInfo
├── config.py                LayerConfig, EncoderConfig, DecoderConfig
├── layers.py                TransformerLayer (pre-norm residual)
├── encoder.py               Encoder
├── decoder.py               DecoderLayer, Decoder
├── cross_attender.py        CrossAttenderLayer, CrossAttenderStack
├── attention/
│   ├── config.py            AttentionConfig
│   ├── protocols.py         AttnKernel, AttnBiasBuilder, SelfAttn, CrossAttn
│   ├── bias.py              NoBiasBuilder, ALiBiBuilder
│   ├── bias_config.py       NoBiasConfig, ALiBiConfig, BiasBuilderConfig
│   ├── bias_factory.py      build_bias_builder
│   ├── self_attn.py         SelfAttention
│   ├── cross_attn.py        CrossAttention
│   └── kernels/
│       ├── config.py        SDPAKernelConfig, WindowedSDPAKernelConfig, …, KernelConfig
│       ├── factory.py       build_kernel
│       ├── sdpa.py          SDPAKernel
│       ├── windowed.py      WindowedSDPAKernel
│       ├── varlen.py        VarlenSDPAKernel
│       ├── varlen_windowed.py  VarlenWindowedSDPAKernel
│       └── _mask.py         build_window_mask (internal helper)
├── feedforward/
│   ├── config.py            FeedForwardConfig
│   ├── protocols.py         FeedForward
│   ├── swiglu.py            SwiGLU
│   └── factory.py           build_ff
├── norm/
│   ├── config.py            RMSNormConfig, LayerNormConfig, NormConfig
│   ├── protocols.py         Norm
│   ├── rms.py               RMSNorm
│   └── factory.py           build_norm
├── positional/
│   ├── config.py            YaRNConfig, RoPE1DConfig, NoPosEncodingConfig, PosEncodingConfig
│   ├── protocols.py         PosEncoding, PackedPosEncoding
│   ├── none.py              NoPosEncoding (null object)
│   ├── rope1d.py            RotaryEmbedding1D
│   ├── rope2d.py            RotaryEmbedding2D
│   └── factory.py           build_pos_encoding
└── presets/
    ├── encoder.py           TransformerEncoder, TransformerEncoderConfig
    ├── decoder.py           TransformerDecoder, TransformerDecoderConfig
    └── cross_attender.py    CrossAttender, CrossAttenderConfig
```

---

## Sequence types

```python
from stackformers import make_padded_input, make_packed_input

# Padded batch — mask is True for valid tokens
inp = make_padded_input(x, mask)           # PaddedInput(x, PaddedSequence(mask))

# Packed batch (variable-length, FlashAttention convention)
inp = make_packed_input(x, cu_seqlens, max_seqlen=512)  # PackedInput(x, PackedSequence(...))
```

`SequenceInfo = PaddedSequence | PackedSequence` is a sealed union. New sequence types are new dataclasses — existing variants are never modified.

---

## Attention kernels

| Kernel | Sequence type | Use case |
|--------|--------------|---------|
| `SDPAKernel` | padded | Default; `F.scaled_dot_product_attention` |
| `WindowedSDPAKernel` | padded | Sliding-window local attention; pure PyTorch mask, no extra deps |
| `VarlenSDPAKernel` | packed | Full attention; `varlen_attn` on CUDA+fp16, loop fallback on CPU |
| `VarlenWindowedSDPAKernel` | packed | Sliding-window local attention; `varlen_attn` with finite window on CUDA+fp16, loop fallback on CPU |

All kernels are pure PyTorch — no optional third-party installs required.

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
just test-cov    pytest + coverage
just check       fmt-check + lint + types + test  (CI gate)
just fix         fmt + lint-fix, then check
just clean       remove build artifacts
```

---

## Roadmap

| Area | Status | Notes |
|------|--------|-------|
| `sequence.py` — PaddedSequence, PackedSequence, PaddedInput, PackedInput | ✅ done | Sealed union, frozen dataclasses |
| `*/protocols.py` — PosEncoding, AttnBiasBuilder, AttnKernel, Norm | ✅ done | Per-module, `@runtime_checkable` |
| `*/config.py` — Pydantic models with `kind` discriminators | ✅ done | AttentionConfig, FeedForwardConfig, NormConfig, PosEncodingConfig, KernelConfig, BiasBuilderConfig |
| `*/factory.py` — per-component builder functions | ✅ done | build_norm, build_ff, build_pos_encoding, build_kernel, build_bias_builder |
| `norm/rms.py` — RMSNorm | ✅ done | |
| `positional/none.py` — NoPosEncoding | ✅ done | Null object for padded + packed protocols |
| `positional/rope1d.py` — RotaryEmbedding1D | ✅ done | Halved-convention; YaRN context extension |
| `positional/rope2d.py` — RotaryEmbedding2D | ✅ done | Row/col split |
| `attention/bias.py` — NoBiasBuilder, ALiBiBuilder | ✅ done | |
| `attention/kernels/` — SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel, VarlenWindowedSDPAKernel | ✅ done | One file per kernel; pure PyTorch, no extra deps |
| `attention/self_attn.py` — SelfAttention (MHA / GQA / MQA) | ✅ done | |
| `attention/cross_attn.py` — CrossAttention | ✅ done | |
| `feedforward/swiglu.py` — SwiGLU | ✅ done | |
| `layers.py` — TransformerLayer | ✅ done | Pre-norm residual |
| `encoder.py` — Encoder | ✅ done | |
| `decoder.py` — DecoderLayer, Decoder | ✅ done | |
| `cross_attender.py` — CrossAttenderLayer, CrossAttenderStack | ✅ done | |
| `presets/` — TransformerEncoder, TransformerDecoder, CrossAttender | ✅ done | Kernel and bias builder fully config-driven |
| FlexAttention kernel | 🔲 planned | `torch.nn.attention.flex_attention` |
| MLA (Multi-Latent Attention) | 🔲 planned | Latent Q/KV projections |
| Sparse / mixture-of-experts FFN | 🔲 planned | Drop-in SwiGLU replacement |
| KV-cache support | 🔲 planned | Incremental decode path |
| `beartype` integration in tests | 🔲 planned | Runtime shape contract checking |

---

## License

See [LICENSE](LICENSE).
