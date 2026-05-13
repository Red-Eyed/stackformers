# norm

Normalisation layers.

## Files

| File | Contents |
|------|----------|
| `config.py` | `RMSNormConfig(dim)`, `LayerNormConfig(dim, eps)`; union `NormConfig` |
| `protocols.py` | `Norm` — `(x: b n d) → b n d` |
| `rms.py` | `RMSNorm` — scales by learned `g`, no centering; numerically stable on fp16 |
| `factory.py` | `build_norm(config: NormConfig) -> Norm` — dispatches on `kind` |

## Adding a new norm

1. Add its config to `config.py` with a `kind: Literal["your_kind"]` field and include it in the `NormConfig` union.
2. Implement the class; satisfy `Norm` structurally (no import of the protocol).
3. Add a `case NewNormConfig()` branch in `factory.py::build_norm`.
