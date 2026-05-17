# CLAUDE.md — stackformers

## What this project is
Typed, composable, SOLID transformer library for PyTorch.
Behavior comes from injected dependencies, not constructor flags.

## Key rules — never violate these

- No optional fields on dataclasses to encode different modes — use sealed unions instead
- No `if self.x is not None` inside `forward()` — use null objects (e.g. `NoPosEncoding`)
- No flags in `__init__` that change `forward()` behavior — new behavior = new class
- `forward()` must be traceable by torch.compile and torch.export — no Python control flow on tensors
- One concept per file — if you can't describe it in one sentence, split it
- All tensor args annotated with jaxtyping shape strings using project dim vocabulary
- Every collaborator accepted by `__init__` must be typed as a protocol, never as a concrete class

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

## File organisation

- All code lives directly under `stackformers/`. Breaking changes go in a new top-level package.
- Configs live next to the class they configure: `attention/config.py`, `feedforward/config.py`.
- Cross-cutting configs (`LayerConfig`, `EncoderConfig`, `DecoderConfig`) live at `config.py`.
- Each module has a `factory.py` with a `build_*` function that dispatches on config type via `match`.
- Opinionated presets live in `presets/` — one file per preset class.
- Protocols are defined in the module they belong to (`attention/protocols.py`, etc.), not a central file.
- Implementations satisfy protocols structurally — they never import the protocol they implement.

## Config and factory conventions

Every union config type uses a `kind: Literal[...]` discriminator field so configs round-trip through JSON unambiguously. The union itself is an `Annotated[..., Field(discriminator="kind")]` alias defined in the same `config.py`.

Each module owns its builder in a co-located `factory.py`. The function signature is `build_*(config: *Config, ...) -> Protocol`. Presets call these factories — they never instantiate concrete classes directly.

## Module-level READMEs

Each subdirectory has a `README.md`. When you change a module's design — add a variant, remove a concept, change a dispatch strategy — update its `README.md` in the same commit. The README covers: purpose in one or two sentences, key design decisions, and how to extend with a new variant. It must not restate what reading the source already gives you (no file tables, no copied signatures).

## Testing rules
- One test file per source file, mirroring directory structure
- Use pytest fixtures for module construction, inputs, and sequence objects — no plain helper functions
- Parametrise over `device_dtype` (all device×dtype combos) for compute tests; use `device` alone (float32) for gradient tests
- Test shape contracts, not numeric values
- Numerical assertions use `atol(dtype)` from `tests/conftest.py`

## Dependencies
- PyTorch native first: `F.scaled_dot_product_attention`, `torch.nn.attention.varlen.varlen_attn`
- No optional third-party kernel dependencies — all kernels are pure PyTorch
- Pydantic for configs; use `Field(gt=0)` / `Field(ge=0)` constraints, not `@field_validator`, for simple bounds
- Prefer `nn.Module` activations (`nn.SiLU()`, `nn.GELU()`) over `torch.nn.functional` calls

## x-transformers reference
Cloned at `./x-transformers/` — read for math and implementation details only.
Do not copy its architecture.

## Versioning

Version is maintained in `pyproject.toml` under `[project].version` and exposed at runtime via `stackformers.__version__`.

Use [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking public API change
- **MINOR** — new backwards-compatible feature
- **PATCH** — bug fix or internal change with no API impact

Bump the version in `pyproject.toml` on every commit that changes behaviour. Commits that only touch docs, tests, or CI do not require a bump.

## Commands
```bash
just fmt          # ruff format
just lint         # ruff check
just types        # pyrefly
just test         # pytest
just check        # full CI gate: fmt-check + lint + types + test
```

## PyTorch
- Before writing any torch code, run: `python -c "import torch; print(torch.__version__)"`
- Always prefer built-in torch implementations over custom ones
- Check `.venv/lib/python3.*/site-packages/torch/nn/modules/` for what's available
