# CLAUDE.md — stackformers

## What this project is
Typed, composable, SOLID transformer library for PyTorch.
Behavior comes from injected dependencies, not constructor flags.

## Key rules — never violate these

- No optional fields on dataclasses to encode different modes — use sealed unions instead
- No `if self.x is not None` inside `forward()` — use null objects (NoPosEncoding, NoBiasBuilder)
- No flags in `__init__` that change `forward()` behavior — new behavior = new class
- `forward()` must be traceable by torch.compile and torch.export — no Python control flow on tensors
- One concept per file — if you can't describe it in one sentence, split it
- All tensor args annotated with jaxtyping shape strings using project dim vocabulary

## Protocols

Every collaborator accepted by `__init__` must be typed as a protocol, never as a concrete class.

```python
# wrong
def __init__(self, norm: RMSNorm, ff: SwiGLU): ...

# right
def __init__(self, norm: Norm, ff: FeedForward): ...
```

### Two kinds of protocols

**High-level** — called at the `nn.Module` call site via `()`.
Declare `__call__`, not `forward`. This is what type checkers need to recognise them as callable.

| Protocol | `__call__` signature | Implementations |
|----------|---------------------|-----------------|
| `Norm` | `(x: b n d) → b n d` | `RMSNorm` |
| `FeedForward` | `(x: b n d) → b n d` | `SwiGLU` |
| `SelfAttn` | `(x: b n d, seq_info) → b n d` | `SelfAttention` |
| `CrossAttn` | `(x: b n d, context: b s d, ctx_seq_info?) → b n d` | `CrossAttention` |

**Low-level** — called explicitly via `.forward()` inside another module's `forward`.
Declare `forward`, not `__call__`.

| Protocol | `forward` signature | Implementations |
|----------|---------------------|-----------------|
| `AttnKernel` | `(q, k, v, attn_mask, attn_bias, is_causal) → b h n dh` | `SDPAKernel`, `WindowedSDPAKernel`, … |
| `AttnBiasBuilder` | `(n, s, device) → h n s \| None` | `ALiBiBuilder`, `NoBiasBuilder` |
| `PosEncoding` | `(q: b h n dh, k: b h s dh) → (q, k)` | `RotaryEmbedding1D`, `NoPosEncoding` |
| `PackedPosEncoding` | `(q: nt h dh, k, position_ids) → (q, k)` | `RotaryEmbedding1D`, `NoPosEncoding` |

### Protocol placement

Define each protocol in the module it most naturally belongs to — not in a central `protocols.py`.

| Protocol file | Contains |
|--------------|---------|
| `attention/protocols.py` | `AttnKernel`, `AttnBiasBuilder`, `SelfAttn`, `CrossAttn` |
| `positional/protocols.py` | `PosEncoding`, `PackedPosEncoding` |
| `feedforward/protocols.py` | `FeedForward` |
| `norm/protocols.py` | `Norm` |

Implementations satisfy protocols structurally — they never import the protocol they implement.

## Dim naming convention (use everywhere)
| Symbol | Meaning |
|--------|---------|
| b      | batch size |
| n      | query sequence length |
| s      | key/value sequence length (= n for self-attn) |
| d      | model dim |
| h      | number of heads |
| dh     | dim per head |
| w      | window size |
| nt     | total tokens in packed sequence |

## SequenceInfo is a sealed union
```python
SequenceInfo = PaddedSequence | PackedSequence
```
New sequence types = new dataclass. Never add optional fields to existing variants.

## File organisation

- All code lives directly under `stackformers/`. Breaking changes go in a new top-level package.
- Configs live next to the class they configure: `attention/config.py`, `feedforward/config.py`.
- Cross-cutting configs (`LayerConfig`, `EncoderConfig`, `DecoderConfig`) live at `config.py`.
- Kernel variants live in `attention/kernels/` — one file per kernel class.
- Opinionated presets live in `presets/` — one file per preset class.

## Presets

`presets/` contains ready-to-use `nn.Module` subclasses that wire up building blocks with fixed choices (RMSNorm + SwiGLU + RoPE-1D + SDPA). Each preset is generic over its config type so subclasses can extend the config and keep full type safety:

| Preset | Config | Notes |
|--------|--------|-------|
| `TransformerEncoder` | `TransformerEncoderConfig` | self-attn stack; `causal=True` for GPT-style |
| `TransformerEncoderCross` | `TransformerEncoderCrossConfig` | self-attn + cross-attn; `context_dim == dim` |

Presets are intentionally not flexible — for custom wiring use the building blocks directly.

## Testing rules
- One test file per source file, mirroring directory structure
- Use pytest fixtures for module construction, inputs, and sequence objects — no plain helper functions
- Parametrise over `device_dtype` (all device×dtype combos) for compute tests; use `device` alone (float32) for gradient tests
- Test shape contracts, not numeric values
- Numerical assertions (norm preservation etc.) use `atol(dtype)` from `tests/conftest.py`

## Dependencies
- PyTorch native first: `F.scaled_dot_product_attention`, `torch.nn.attention.varlen.varlen_attn`
- No optional third-party kernel dependencies — all kernels are pure PyTorch
- Pydantic for configs; use `Field(gt=0)` / `Field(ge=0)` constraints, not `@field_validator`, for simple bounds

## x-transformers reference
Cloned at `./x-transformers/` — read for math and implementation details only.
Do not copy its architecture.

## Commands
```bash
# Install (uv)
uv sync --group dev

# Lint / format
just lint
just fmt

# Type-check
just types        # runs pyrefly

# Tests
just test
just check        # full CI gate: fmt-check + lint + types + test
```
