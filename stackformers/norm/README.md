# norm

Normalisation layers.

## Files

| File | Contents |
|------|----------|
| `config.py` | `RMSNormConfig(dim)`, `LayerNormConfig(dim, eps)` |
| `protocols.py` | `Norm` — `(x: b n d) → b n d` |
| `rms.py` | `RMSNorm` — scales by learned `g`, no centering; numerically stable on fp16 |

## Adding a new norm

1. Add its config to `config.py`.
2. Implement the class; satisfy `Norm` structurally (no import of the protocol).
3. Add a `case NewNormConfig()` branch in `presets/configs.py::build_norm`.
