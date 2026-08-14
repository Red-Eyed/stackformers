# Feed-forward Development Log

## 2026-08-14 — Standard GELU FFN

### Motivation

The module exposed gated SwiGLU and GEGLU networks and a non-gated ReLU² network, but not the
standard GELU Transformer FFN used as the parameter-count baseline in their documentation.

### Decision

Add `GELUFFN` as a distinct, bias-free implementation selected by `GELUConfig(kind="gelu")`.
It uses the standard non-gated hidden width `int(dim * mult)`, PyTorch's built-in
`nn.GELU(approximate="tanh")`, dropout after activation, and a second projection back to the
model dimension. Existing presets remain on SwiGLU; selecting GELU is explicit through injected
configuration.

### Verified result

Ruff, formatting, and Pyrefly checks pass. The full suite passes with 281 tests and 19 skips;
focused device/dtype-expanded tests verify the output shape, bias-free projections, hidden
width, GELU approximation, input gradients, config round-tripping, and factory dispatch.

### Unproven claims

No training-quality, convergence, throughput, or memory comparison with the other feed-forward
variants has been run.
