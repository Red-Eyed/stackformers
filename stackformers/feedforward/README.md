# feedforward

Feed-forward sublayers applied token-wise after attention.

## Files

| File | Contents |
|------|----------|
| `config.py` | `FeedForwardConfig(dim, mult, dropout)` |
| `protocols.py` | `FeedForward` — `(x: b n d) → b n d` |
| `swiglu.py` | `SwiGLU` — gated linear unit with SiLU activation; inner_dim = `int(dim * mult * 2/3)` rounded to a multiple of 64 |

## Adding a new feed-forward

1. Add its config to `config.py` (or reuse `FeedForwardConfig` if parameterisation is identical).
2. Implement the class; satisfy `FeedForward` structurally.
3. Add a `case` branch in `presets/configs.py::build_ff` (currently a direct call — convert to `match` when a second variant exists).
