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

## Projection layout

Select concrete config variants to construct packed projections directly:

```python
config = plain_encoder_config(dim=768, heads=12, num_layers=6)
config.attn = PackedSelfAttentionConfig(dim=768, heads=12, dim_head=64)
config.ff = PackedSwiGLUConfig(dim=768)
model = TransformerEncoder(config)

saved_config = config.model_dump_json()
restored_config = TransformerEncoderConfig.model_validate_json(saved_config)
restored = TransformerEncoder(restored_config)
restored.load_state_dict(model.state_dict())
```

Decoder configs accept `PackedSelfAttentionConfig` and `PackedCrossAttentionConfig` independently.
Cross-attender configs accept the cross-attention variant; variable-width encoders select the
variant in each layer. Feed-forward selection is independent of attention selection.
Defaults remain ordinary projections. Old attention JSON without a `kind` discriminator also
selects ordinary projections, preserving old checkpoint layouts. For existing weights, load
the ordinary checkpoint before calling `to_packed()` on the supported modules; save a matching
packed preset config alongside the converted state dict. Alternatively, call the public
`model = to_packed(model)` helper to replace layers recursively and update built-in preset
configs together. `to_unpacked(model)` reverses both changes. Config updates use copies, so
external references to the original configs remain valid. Custom preset subclasses retain
their own metadata. A state dict alone carries no config.
Existing cached decoding wrappers require ordinary attention.

## Per-call attention visibility

Encoder presets (including the variable-width encoder) accept `self_attn_mask`. Decoder
presets accept both `self_attn_mask` and `cross_attn_mask`; CrossAttender accepts only
`cross_attn_mask`. Layer and stack APIs expose the same arguments.

```python
encoder(input, self_attn_mask="causal")
decoder(text, image_memory, self_attn_mask="causal")
decoder(text, self_attn_mask="bidirectional")
decoder(text, image_memory, self_attn_mask=text_visibility, cross_attn_mask=image_visibility)
```

`self_attn_mask="default"` uses each layer's initialization settings. `"causal"`,
`"bidirectional"`, or a boolean `(n, n)` tensor replaces those causal/window settings.
`cross_attn_mask=None` allows every valid context token; a boolean `(n, s)` tensor restricts
query/context pairs. True means allowed. Masks share a pattern across documents and heads;
padding and packed document boundaries always remain enforced. For packed inputs, dimensions
are `max_seqlen`, and coordinates restart within each document. Tensor masks must be on the
input device and use padded SDPA fallback; named modes retain the packed fast path.
A cross mask without context raises `ValueError`.

Named modes specialize during compilation/export; tensor masks remain runtime graph inputs.
Cached generation retains its existing causal API and does not accept these overrides.
Custom layer subclasses must accept and forward the new keyword arguments. Custom attention
collaborators need them when a caller requests an override.

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
