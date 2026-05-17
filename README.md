# stackformers

Typed, composable transformer library for PyTorch.
Every architectural choice — positional encoding, normalization, feedforward variant — is an injected dependency, not a constructor flag.

```bash
uv add stackformers
```

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

Encoder–decoder:

```python
from stackformers import TransformerDecoder, plain_decoder_config

model = TransformerDecoder(plain_decoder_config(dim=512, heads=8, num_layers=6))
out   = model(make_padded_input(x, mask), make_padded_input(context, ctx_mask))
```

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
| Feedforward | SwiGLU, GEGLU |
| Normalization | RMSNorm, LayerNorm |
| Presets | Encoder, Decoder, CrossAttender |

On CUDA with fp16/bf16 the packed path uses `torch.nn.attention.varlen.varlen_attn`. CPU and fp32 fall back to a per-sequence SDPA loop — correct everywhere, fast where it matters.

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
