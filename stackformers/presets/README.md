# presets

Opinionated, ready-to-use `nn.Module` subclasses. Each preset wires up building blocks with fixed structural choices (RMSNorm, SwiGLU, SDPA) while letting the caller control all hyperparameters through a typed config.

Presets are intentionally not flexible. For custom wiring, use the building blocks directly.

## Preset comparison

| Preset | Per-layer ops | `forward` inputs | Use case |
|--------|--------------|-----------------|----------|
| `TransformerEncoder` | self-attn → ff | `x`, `seq_info` | Bidirectional encoder; set `causal=True` for GPT-style decoder |
| `TransformerEncoderCross` | self-attn → cross-attn → ff | `x`, `context`, `seq_info`, `ctx_seq_info?` | x has its own sequence structure and attends to a context (e.g. encoder-decoder) |
| `CrossAttender` | cross-attn → ff | `x`, `context`, `ctx_seq_info?` | x is a set of queries with no self-attention (e.g. Perceiver, slot attention, learned queries) |

`TransformerEncoderCross` internally reuses `Decoder`/`DecoderLayer` (the mid-level blocks that implement self-attn + cross-attn + ff). The preset name reflects its role — not its implementation ancestry.

## Config composition

Each preset config is built from sub-configs that belong to the primitive they configure:

```
TransformerEncoderConfig
├── attn: AttentionConfig        (attention/config.py)
├── ff: FeedForwardConfig        (feedforward/config.py)
├── norm: RMSNormConfig | LayerNormConfig   (norm/config.py)
├── pos_encoding: RoPE1DConfig | NoPosEncodingConfig  (positional/config.py)
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
