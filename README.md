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

```python
from stackformers import (
    AttentionConfig, FeedForwardConfig, EncoderConfig, LayerConfig,
    build_encoder, build_gpt,
    make_padded,
)
import torch

# GPT-style causal backbone (RoPE + SDPA, no extra config)
model = build_gpt(dim=768, heads=12, dim_head=64, num_layers=12)

x = torch.randn(2, 128, 768)
mask = torch.ones(2, 128, dtype=torch.bool)
out = model(x, make_padded(mask))   # (2, 128, 768)
```

### Custom encoder — swap every piece

```python
from stackformers import (
    AttentionConfig, FeedForwardConfig, LayerConfig, EncoderConfig,
    SelfAttention, SwiGLU, TransformerLayer, Encoder, RMSNorm,
    RotaryEmbedding1D, ALiBiBuilder, SDPAKernel,
    make_padded,
)

attn_cfg = AttentionConfig(dim=512, heads=8, dim_head=64, causal=False)
ff_cfg   = FeedForwardConfig(dim=512, mult=4.0)

layers = [
    TransformerLayer(
        self_attn=SelfAttention(
            config=attn_cfg,
            pos_encoding=RotaryEmbedding1D(dim_head=64),
            bias_builder=ALiBiBuilder(heads=8),
            kernel=SDPAKernel(),
        ),
        ff=SwiGLU(ff_cfg),
        norm_attn=RMSNorm(512),
        norm_ff=RMSNorm(512),
    )
    for _ in range(6)
]
encoder = Encoder(layers=layers, final_norm=RMSNorm(512))
```

---

## Module map

```
stackformers/v1/
├── sequence.py          PaddedSequence, PackedSequence, SequenceInfo
├── protocols.py         PosEncoding, AttnBiasBuilder, AttnKernel, Norm
├── configs.py           Pydantic configs with validators
├── layers.py            TransformerLayer (pre-norm residual)
├── encoder.py           Encoder
├── decoder.py           DecoderLayer, Decoder
├── factories.py         build_encoder(), build_decoder(), build_gpt()
├── attention/
│   ├── bias.py          NoBiasBuilder, ALiBiBuilder
│   ├── kernels.py       SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel
│   ├── kernels_flash.py FlashVarlenKernel (optional; requires flash-attn)
│   ├── self_attn.py     SelfAttention
│   └── cross_attn.py   CrossAttention
├── feedforward/
│   └── swiglu.py        SwiGLU
├── norm/
│   └── rms.py           RMSNorm
└── positional/
    ├── none.py          NoPosEncoding (null object)
    ├── rope1d.py        RotaryEmbedding1D
    └── rope2d.py        RotaryEmbedding2D
```

---

## Sequence types

```python
from stackformers import PaddedSequence, PackedSequence, make_padded, make_packed

# Padded batch — mask is True for valid tokens
seq = make_padded(mask)                          # Bool[Tensor, "b n"]

# Packed batch (variable-length, FlashAttention convention)
seq = make_packed(cu_seqlens, max_seqlen=512)    # Int[Tensor, "b+1"], int
```

`SequenceInfo = PaddedSequence | PackedSequence` is a sealed union. New sequence types are new dataclasses — existing variants are never modified.

---

## Attention kernels

| Kernel | Use case |
|--------|----------|
| `SDPAKernel` | Default; uses `F.scaled_dot_product_attention` (padded batches) |
| `VarlenSDPAKernel` | Packed sequences; loops over variable-length items |
| `WindowedSDPAKernel` | Local sliding-window; falls back to full SDPA when window ≥ seq len |
| `FlashVarlenKernel` | Packed sequences via `flash_attn_varlen_func` (optional install) |

Third-party kernels are never imported at module load time — only inside `__init__`.

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

Optional flash-attn support:

```bash
pip install flash-attn --no-build-isolation
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
| `sequence.py` — PaddedSequence, PackedSequence | ✅ done | Sealed union, frozen dataclasses |
| `protocols.py` — PosEncoding, AttnBiasBuilder, AttnKernel, Norm | ✅ done | `@runtime_checkable` |
| `configs.py` — Pydantic models + validators | ✅ done | AttentionConfig, FeedForwardConfig, LayerConfig, EncoderConfig, DecoderConfig |
| `norm/rms.py` — RMSNorm | ✅ done | |
| `positional/none.py` — NoPosEncoding | ✅ done | Null object |
| `positional/rope1d.py` — RotaryEmbedding1D | ✅ done | Halved-convention, norm-preserving |
| `positional/rope2d.py` — RotaryEmbedding2D | ✅ done | Row/col split |
| `attention/bias.py` — NoBiasBuilder, ALiBiBuilder | ✅ done | |
| `attention/kernels.py` — SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel | ✅ done | |
| `attention/kernels_flash.py` — FlashVarlenKernel | ✅ done | Lazy import; requires flash-attn |
| `attention/self_attn.py` — SelfAttention (MHA / GQA / MQA) | ✅ done | |
| `attention/cross_attn.py` — CrossAttention | ✅ done | |
| `feedforward/swiglu.py` — SwiGLU | ✅ done | |
| `layers.py` — TransformerLayer | ✅ done | Pre-norm residual |
| `encoder.py` — Encoder | ✅ done | |
| `decoder.py` — DecoderLayer, Decoder | ✅ done | |
| `factories.py` — build_encoder, build_decoder, build_gpt | ✅ done | |
| `v2/` — breaking-change namespace | 🔲 planned | Trigger: first API-breaking change after v1 tag |
| FlexAttention kernel | 🔲 planned | `torch.nn.attention.flex_attention` |
| MLA (Multi-Latent Attention) | 🔲 planned | Latent Q/KV projections |
| Sparse / mixture-of-experts FFN | 🔲 planned | Drop-in SwiGLU replacement |
| KV-cache support | 🔲 planned | Incremental decode path |
| `beartype` integration in tests | 🔲 planned | Runtime shape contract checking |

---

## Versioning

All stable code lives under `stackformers/v1/`. When breaking changes are needed a new `v2/` package is created. `v1/` is frozen once tagged — its `__init__.py` becomes read-only.

---

## License

See [LICENSE](LICENSE).
