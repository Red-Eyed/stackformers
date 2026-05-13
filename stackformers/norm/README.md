# norm

Normalisation layers behind the `Norm` protocol: `(x: b n d) → b n d`.

`RMSNorm` is the default — no mean subtraction, no bias, numerically stable on fp16. `LayerNorm` is available for compatibility.

## Adding a new norm

1. Add a config class with a `kind: Literal[...]` discriminator to `config.py` and include it in the `NormConfig` union.
2. Implement the class satisfying `Norm` structurally — do not import the protocol.
3. Add a `case` branch in `factory.py::build_norm`.
