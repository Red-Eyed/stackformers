# presets

Opinionated, ready-to-use `nn.Module` subclasses. Each preset wires up building blocks with fixed structural choices while letting the caller control all hyperparameters and implementation variants through a typed config.

Presets are intentionally not flexible. For custom wiring, use the building blocks directly.

## Preset comparison

| Preset | Per-layer ops | `forward` inputs | Use case |
|--------|--------------|-----------------|----------|
| `TransformerEncoder` | self-attn → ff | `SequenceInput` | Bidirectional or causal encoder; set `causal=True` in `attn` for GPT-style |
| `TransformerDecoder` | causal self-attn → cross-attn → ff | `x_input`, `ctx_input` | Target sequence attends to context; self-attention is always causal |
| `CrossAttender` | cross-attn → ff | `x_input`, `ctx_input` | x is a fixed set of queries with no self-attention (e.g. Perceiver, slot attention) |

`SequenceInput = PaddedInput | PackedInput`. All presets accept either variant — the underlying kernels and pos-encodings dispatch internally via `match`.

## Config composition

Each preset config is built from sub-configs that belong to the primitive they configure:

```
TransformerEncoderConfig
├── attn: AttentionConfig                                    (attention/config.py)
├── ff: FeedForwardConfig                                    (feedforward/config.py)
├── norm: RMSNormConfig | LayerNormConfig                    (norm/config.py)
├── pos_encoding: RoPE1DConfig | NoPosEncodingConfig         (positional/config.py)
├── kernel: SDPAKernelConfig | WindowedSDPAKernelConfig | …  (attention/kernels/config.py)
├── bias: NoBiasConfig | ALiBiConfig                         (attention/bias_config.py)
└── num_layers: int

TransformerDecoderConfig
├── self_attn: AttentionConfig         (causal=True enforced internally)
├── cross_attn: AttentionConfig        (causal=False enforced internally)
├── ff: FeedForwardConfig
├── norm: RMSNormConfig | LayerNormConfig
├── pos_encoding: RoPE1DConfig | NoPosEncodingConfig  (self-attn only; cross-attn uses NoPosEncoding)
├── self_attn_kernel: KernelConfig
├── self_attn_bias: BiasBuilderConfig
├── cross_attn_kernel: KernelConfig
├── cross_attn_bias: BiasBuilderConfig
└── num_layers: int

CrossAttenderConfig
├── attn: AttentionConfig              (causal=False enforced internally)
├── ff: FeedForwardConfig
├── norm: RMSNormConfig | LayerNormConfig
├── kernel: KernelConfig
├── bias: BiasBuilderConfig
└── num_layers: int
```

All union config fields are discriminated on `kind` and serialise cleanly to/from JSON.

## Builder functions

Each component owns its construction logic in a co-located `factory.py`:

| Factory | Function | Lives in |
|---------|----------|----------|
| `build_norm` | `NormConfig → Norm` | `norm/factory.py` |
| `build_ff` | `FeedForwardConfig → FeedForward` | `feedforward/factory.py` |
| `build_pos_encoding` | `PosEncodingConfig → PosEncoding` | `positional/factory.py` |
| `build_kernel` | `KernelConfig, causal, dropout → AttnKernel` | `attention/kernels/factory.py` |
| `build_bias_builder` | `BiasBuilderConfig, heads, causal → AttnBiasBuilder` | `attention/bias_factory.py` |

## Extending a preset

Subclass the config to add fields, then subclass the preset and override `__init__`:

```python
class MyEncoderConfig(TransformerEncoderConfig):
    extra_field: int = 4

class MyEncoder(TransformerEncoder):
    def __init__(self, config: MyEncoderConfig) -> None:
        super().__init__(config)
        self.config: MyEncoderConfig  # narrow the type for IDE support
        # use config.extra_field
```

## Padded vs packed sequences

The same preset works for both training (packed) and inference (padded) — no code changes required. Swap the kernel config and pass the appropriate `SequenceInput`:

- `make_padded_input(x, mask)` → `PaddedInput` — standard batched tensors with a boolean key-padding mask
- `make_packed_input(x, cu_seqlens, max_seqlen)` → `PackedInput` — FlashAttention-style flat pack; use with varlen kernels for training without padding waste

The attention weight matrices (`to_q`, `to_k`, `to_v`, `to_out`) are shared between padded and packed paths, so `load_state_dict` transfers directly.
