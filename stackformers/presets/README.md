# presets

Opinionated `nn.Module` subclasses that wire up the building blocks with fixed structural choices. The caller controls all hyperparameters through a typed config — nothing is hardcoded.

For custom wiring, use the building blocks directly.

## Class hierarchy

Each preset is split into two levels:

- **`*Base[C]`** — abstract generic class parameterised over any config type `C`. Defines `__init__` (wires the stack) and `forward`, declares `build_layers` and `build_norm` as abstract. No assumptions about `C`.
- **`*`** (concrete) — extends `*Base[ConcreteConfig]`, fills in all `build_*` defaults, and exposes additional overridable hooks (`build_pos_encoding`, `build_attn_bias`, `build_ff`, …).

## Presets

**`TransformerEncoder`** — self-attn → ff per layer. Set `causal=True` in `attn` for GPT-style autoregressive decoding. Set `window_size` for sliding-window (O(n·w)) attention.

**`TransformerDecoder`** — causal self-attn → cross-attn → ff per layer. Cross-attention always uses no positional encoding. Call `decoder(target, context)` for
conditioned decoding, or `decoder(target)` / `decoder(target, None)` to skip the complete
cross-attention branch, including its normalization, in every layer. Cross-attention
parameters remain registered so the same model supports either call.

For cached decoding, pass `None` as the cross cache:
`cached_decoder(token, None, self_kv_cache, step_i)`. No context cache builder is needed.
Keep context fixed throughout a cached sequence; when changing between conditioned and
unconditioned decoding, rebuild the self cache from the prefix.
Compilation and export specialize to the context presence used when tracing.

**`CrossAttender`** — cross-attn → ff per layer. `x` is a fixed set of queries with no self-attention. Useful for Perceiver-style or slot-attention architectures.

All presets accept `PaddedInput` or `PackedInput` interchangeably.

## Customising a collaborator (subclass the concrete preset)

Override one or more `build_*` methods; the rest use their defaults:

```python
class ALiBiEncoder(TransformerEncoder):
    def build_attn_bias(self, _: TransformerEncoderConfig) -> AttnBias:
        return ALiBi(self.config.attn.heads)
```

## Using a fully custom config (subclass the abstract base)

Implement `build_layers` and `build_norm`; the base `__init__` calls them and nothing else:

```python
class MyConfig(BaseModel):
    dim: int
    heads: int
    num_layers: int


class MyEncoder(TransformerEncoderBase[MyConfig]):
    def build_layers(self, config: MyConfig) -> list[TransformerLayer]:
        pos = MyRoPE(config.dim // config.heads)
        return [
            TransformerLayer(
                self_attn=SelfAttention(
                    SelfAttentionConfig(
                        dim=config.dim, heads=config.heads, dim_head=config.dim // config.heads
                    ),
                    pos_encoding=pos,
                ),
                ff=SwiGLU(SwiGLUConfig(dim=config.dim, mult=4)),
                norm_attn=nn.RMSNorm(config.dim),
                norm_ff=nn.RMSNorm(config.dim),
            )
            for _ in range(config.num_layers)
        ]

    def build_norm(self, config: MyConfig) -> Norm:
        return nn.RMSNorm(config.dim)
```
