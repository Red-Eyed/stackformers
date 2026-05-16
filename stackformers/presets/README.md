# presets

Opinionated `nn.Module` subclasses that wire up the building blocks with fixed structural choices. The caller controls all hyperparameters through a typed config — nothing is hardcoded.

For custom wiring, use the building blocks directly.

## Presets

**`TransformerEncoder`** — self-attn → ff per layer. Set `causal=True` in `attn` for GPT-style autoregressive decoding. Set `window_size` for sliding-window (O(n·w)) attention.

**`TransformerDecoder`** — causal self-attn → cross-attn → ff per layer. Set `causal=True` in `self_attn`. Cross-attention always uses no positional encoding.

**`CrossAttender`** — cross-attn → ff per layer. `x` is a fixed set of queries with no self-attention. Useful for Perceiver-style or slot-attention architectures.

All presets accept `PaddedInput` or `PackedInput` interchangeably — pass the appropriate input type without changing the model.

## Extending a preset

Subclass the config to add fields, then subclass the preset and override `__init__`:

```python
class MyEncoderConfig(TransformerEncoderConfig):
    extra_field: int = 4

class MyEncoder(TransformerEncoder):
    def __init__(self, config: MyEncoderConfig) -> None:
        super().__init__(config)
        self.config: MyEncoderConfig
```
