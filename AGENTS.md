# Repository guidance

## What this project is

Typed, composable, SOLID transformer library for PyTorch.
Behavior comes from injected dependencies, not constructor flags.

## Repository commands

Read `Justfile` before running setup, build, formatting, linting, type-checking, testing, or
cleanup commands. Prefer its existing recipes so local and automated workflows use the same
tool arguments and defaults.

Always use `uv` for Python commands.

```bash
just fmt          # ruff format
just lint         # ruff check
just types        # pyrefly
just test         # pytest
just check        # formatting check, lint, types, and tests
```

## Design rules

- Encode distinct modes with closed unions, not optional fields in structured records.
- Use null objects (e.g. `NoPosEncoding`) instead of `if self.x is not None` inside `forward()`.
- Do not add flags in `__init__` that change `forward()` behavior; introduce a new class.
- Keep `forward()` traceable by `torch.compile` and `torch.export`; avoid Python control flow
  on tensors.
- Keep one concept per file; split files that cannot be described in one sentence.
- Annotate tensor arguments with jaxtyping shape strings using the dimension vocabulary below.
- Type collaborators accepted by `__init__` as protocols rather than concrete implementations.
- Prefer named intermediate values over nested transformation calls or functional pipelines.
  Separate normalization, attention/feed-forward computation, output normalization, and residual
  addition into readable steps so intermediate values are easy to inspect while debugging.
  Keep trivial expressions inline when naming them would add no meaning.

## Python structured records

Never use `dataclasses` in this repository. PyTorch recognizes `NamedTuple` and `TypedDict`
reliably across pytree, export, and compilation boundaries, while dataclasses require additional
registration and handling.

- Use `NamedTuple` for fixed-shape records with named fields, especially tensor inputs and outputs.
- Use `TypedDict` when the value must retain mapping semantics.
- Do not introduce `dataclass`, `@dataclass`, or imports from `dataclasses`.

Ruff enforces the import restriction through `TID251`.

## Expected failures

Use `Result[T, E] = Ok[T] | Err[E]` from `stackformers.result` for recoverable core failures.
The alternatives are generic NamedTuples; do not add an external Result dependency or use
unchecked unwrap methods. Match `Ok(value=value)` / `Err(error=error)` with keyword patterns
for PyTorch tracing compatibility, and use `assert_never` for closed-union exhaustiveness.

Validate admission before construction instead of raising and catching expected failures.
Keep exception policy at public adapters for operations that cannot proceed, preserving
existing API contracts. Catch expected vendor exceptions at the vendor boundary, retaining
their original causes; unexpected programming failures may propagate.

## Dimension vocabulary

| Symbol | Meaning |
|--------|---------|
| `b` | Batch size |
| `n` | Query sequence length |
| `s` | Key/value sequence length (`n` for self-attention) |
| `d` | Model dimension |
| `h` | Number of heads |
| `dh` | Dimension per head |
| `w` | Window size |
| `nt` | Total tokens in a packed sequence |
| `m` | Number of masked tokens gathered across a batch or pack |

## File organization

- All library code lives directly under `stackformers/`. Breaking changes go in a new
  top-level package.
- Keep configs next to the classes they configure, e.g. `attention/config.py` and
  `feedforward/config.py`.
- Keep cross-cutting configs (`LayerConfig`, `EncoderConfig`, `DecoderConfig`) in `config.py`.
- Each module owns a `factory.py` with a `build_*` function that dispatches on config type
  via `match`.
- Keep opinionated presets in `presets/`, with one file per preset class.
- Define protocols in their owning module (`attention/protocols.py`, etc.), not a central file.
- Implement protocols structurally; implementations do not import the protocols they implement.

## Config and factory conventions

Every union config type uses a `kind: Literal[...]` discriminator field so configs round-trip
through JSON unambiguously. Define the union as an
`Annotated[..., Field(discriminator="kind")]` alias in the same `config.py`.

Each module owns its builder in a co-located `factory.py`, with the signature
`build_*(config: *Config, ...) -> Protocol`. Presets call factories instead of instantiating
concrete classes directly.

## Module READMEs

Each module subdirectory has a `README.md`. When changing a module's design by adding a
variant, removing a concept, or changing a dispatch strategy, update its README in the same
commit. Cover its purpose, key design decisions, and how to extend it with a new variant.
Do not repeat information obvious from the source, such as file tables or copied signatures.

## Testing

- Mirror source directory structure, with one test file per source file.
- Use pytest fixtures for module construction, inputs, and sequence objects. Pure transformation
  and assertion helpers may be ordinary functions.
- Parametrize compute tests over `device_dtype` (all device/dtype combinations); use `device`
  alone with float32 for gradient tests.
- Test shape contracts and numerical behavior where correctness or compatibility requires it.
- Use `atol(dtype)` from `tests/conftest.py` for numerical tolerances; exact compatibility
  assertions may require equality instead.

## Dependencies and PyTorch

- Prefer built-in PyTorch implementations over custom ones, especially
  `F.scaled_dot_product_attention` and `torch.nn.attention.varlen.varlen_attn`.
- Do not add optional third-party kernel dependencies; kernels use PyTorch.
- Use Pydantic for configs. Prefer `Field(gt=0)` / `Field(ge=0)` over `@field_validator`
  for simple bounds.
- Prefer `nn.Module` activations (`nn.SiLU()`, `nn.GELU()`) over `torch.nn.functional` calls.
- Before writing PyTorch code, check the installed version with
  `uv run python -c "import torch; print(torch.__version__)"`.
- Check `.venv/lib/python3.*/site-packages/torch/nn/modules/` for available implementations.

## x-transformers reference

Use the local `x-transformers/` checkout, when available, for math and implementation details
only. Do not copy its architecture.

## Versioning

Maintain the version in `pyproject.toml` under `[project].version`, synchronize it with
`uv.lock`, and expose it at runtime through `stackformers.__version__`.

Use Semantic Versioning:

- **MAJOR** for breaking public API changes.
- **MINOR** for new backwards-compatible features.
- **PATCH** for bug fixes or internal changes without API impact.

Bump the version before committing package or behavior changes on `main`. Version bumps on
feature or work branches require an explicit user instruction. Changes limited to repository
guidance, tests, or CI do not require a bump.
