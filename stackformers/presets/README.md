# presets

Opinionated, ready-to-use `nn.Module` subclasses. Each preset wires up building blocks with fixed structural choices (RMSNorm, SwiGLU, SDPA) while letting the caller control all hyperparameters through a typed config.

Presets are intentionally not flexible. For custom wiring, use the building blocks directly.

## Preset comparison

| Preset | Per-layer ops | `forward` inputs | Use case |
|--------|--------------|-----------------|----------|
| `TransformerEncoder` | self-attn → ff | `x`, `seq_info` | Bidirectional or causal encoder; set `causal=True` in `attn` for GPT-style |
| `TransformerDecoder` | causal self-attn → cross-attn → ff | `x`, `context`, `seq_info`, `ctx_seq_info?` | Target sequence attends to context; self-attention is always causal |
| `CrossAttender` | cross-attn → ff | `x`, `context`, `x_seq_info?`, `ctx_seq_info?` | x is a fixed set of queries with no self-attention (e.g. Perceiver, slot attention) |

`seq_info` is `SequenceInfo = PaddedSequence | PackedSequence`. All presets accept either variant — the underlying kernels and pos-encodings dispatch internally via `match`.

## Config composition

Each preset config is built from sub-configs that belong to the primitive they configure:

```
TransformerEncoderConfig
├── attn: AttentionConfig              (attention/config.py)
├── ff: FeedForwardConfig              (feedforward/config.py)
├── norm: RMSNormConfig | LayerNormConfig   (norm/config.py)
├── pos_encoding: RoPE1DConfig | NoPosEncodingConfig  (positional/config.py)
└── num_layers: int

TransformerDecoderConfig
├── self_attn: AttentionConfig         (causal=True enforced internally)
├── cross_attn: AttentionConfig        (causal=False enforced internally)
├── ff: FeedForwardConfig
├── norm: RMSNormConfig | LayerNormConfig
├── pos_encoding: RoPE1DConfig | NoPosEncodingConfig  (self-attn only)
└── num_layers: int

CrossAttenderConfig
├── attn: AttentionConfig              (causal=False enforced internally)
├── ff: FeedForwardConfig
├── norm: RMSNormConfig | LayerNormConfig
└── num_layers: int
```

## Builder functions (`configs.py`)

`build_norm`, `build_ff`, `build_pos_encoding` — dispatch on config type via `match` and return the appropriate module typed as its protocol (`Norm`, `FeedForward`, `PosEncoding`). Add new variants here when new primitives are introduced.

## Extending a preset

Subclass the preset and its config:

```python
class MyEncoderConfig(TransformerEncoderConfig):
    extra_field: int = 4

class MyEncoder(TransformerEncoder[MyEncoderConfig]):
    def __init__(self, config: MyEncoderConfig) -> None:
        super().__init__(config)
        # use config.extra_field
```

`Generic[ConfigT]` ensures `self.config` is typed as the subclass config throughout.

## Padded vs packed sequences

The same preset works for both training (packed) and inference (padded) — no code changes required. Kernels and positional encodings hold a single `forward` that dispatches on the runtime type:

- `PaddedSequence(mask)` — standard batched tensors with a boolean key-padding mask
- `PackedSequence(cu_seqlens, max_seqlen)` — FlashAttention-style flat pack; used with varlen kernels for training without padding waste

To swap between packed training and padded inference, replace the kernel and pass the appropriate `SequenceInfo`. The attention weight matrices (`to_q`, `to_k`, `to_v`, `to_out`) are shared between padded and packed attention subclasses, so `load_state_dict` transfers directly.
