# PLAN.md — stackformer v1

## Goal
Replace x-transformers' flag-driven god objects with clean, injected, protocol-based composition.

## Design principles

1. **Dependency injection over flags** — modules receive collaborators (kernel, pos_encoding, bias_builder) rather than booleans that switch behavior
2. **Sealed unions over optional fields** — `SequenceInfo = PaddedSequence | PackedSequence`
3. **Null objects over None checks** — `NoPosEncoding`, `NoBiasBuilder` implement protocols and do nothing
4. **One concept per file** — atomic decomposition means each file can be understood and tested in isolation
5. **Protocols over ABCs** — structural subtyping via `Protocol` + `@runtime_checkable`

## Module map

```
stackformer/v1/
├── sequence.py        PaddedSequence, PackedSequence, SequenceInfo
├── protocols.py       PosEncoding, AttnBiasBuilder, AttnKernel, Norm
├── configs.py         AttentionConfig, FeedForwardConfig, LayerConfig, EncoderConfig, DecoderConfig
├── layers.py          TransformerLayer (pre-norm residual wrapper)
├── encoder.py         Encoder (stack of TransformerLayers with self-attn)
├── decoder.py         DecoderLayer, Decoder
├── factories.py       build_encoder(), build_decoder(), build_gpt()
├── attention/
│   ├── bias.py        NoBiasBuilder, ALiBiBuilder
│   ├── kernels.py     SDPAKernel, VarlenSDPAKernel, WindowedSDPAKernel
│   ├── kernels_flash.py  FlashVarlenKernel (optional, loaded lazily)
│   ├── self_attn.py   SelfAttention
│   └── cross_attn.py  CrossAttention
├── feedforward/
│   └── swiglu.py      SwiGLU
├── norm/
│   └── rms.py         RMSNorm
└── positional/
    ├── none.py        NoPosEncoding (null object)
    ├── rope1d.py      RotaryEmbedding1D
    └── rope2d.py      RotaryEmbedding2D
```

## Protocols

```python
class PosEncoding(Protocol):
    def forward(self, q, k) -> tuple[Tensor, Tensor]: ...

class AttnBiasBuilder(Protocol):
    def forward(self, n, s, device) -> Tensor | None: ...

class AttnKernel(Protocol):
    def forward(self, q, k, v, attn_mask, attn_bias, is_causal) -> Tensor: ...

class Norm(Protocol):
    def forward(self, x) -> Tensor: ...
```

## SequenceInfo contract

`PaddedSequence.mask` — `Bool[Tensor, "b n"]`, True = valid token
`PackedSequence.cu_seqlens` — `Int[Tensor, "b+1"]` cumulative sequence lengths
`PackedSequence.max_seqlen` — int, maximum individual sequence length

## Attention data flow

```
x: (b, n, d)
  -> norm -> q,k,v projections -> split heads
  -> pos_encoding(q, k) -> (q, k)
  -> bias_builder(n, s) -> bias
  -> kernel(q, k, v, mask, bias, causal) -> out: (b, h, n, dh)
  -> merge heads -> out_proj
  -> residual
```

## Implementation order (each must pass tests before next)
1. sequence.py
2. protocols.py
3. configs.py
4. norm/rms.py
5. positional/none.py
6. positional/rope1d.py
7. positional/rope2d.py
8. attention/bias.py
9. attention/kernels.py
10. attention/self_attn.py
11. attention/cross_attn.py
12. feedforward/swiglu.py
13. layers.py
14. encoder.py
15. decoder.py
16. v1/__init__.py
17. factories.py
18. stackformer/__init__.py

## RoPE math reference (from x-transformers)

```python
inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
freqs = einsum("n, d -> n d", positions, inv_freq)
freqs = cat((freqs, freqs), dim=-1)  # [n, dh]

# rotate_half(x): [-x2, x1] for x = [x1, x2]
out = x * freqs.cos() + rotate_half(x) * freqs.sin()
```

## ALiBi math reference

Slopes for h heads: `m_i = 2^(-8i/h)` for i in 1..h
Bias: `bias[h, i, j] = -m_h * |i - j|` (non-causal) or `-m_h * (i - j)` (causal, j <= i)
