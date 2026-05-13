# CLAUDE.md — stackformer

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

## Versioning
All code lives under `stackformer/v1/`. When breaking changes are needed, create `v2/`.
`v1/` is frozen once tagged. Its `__init__.py` is read-only after that.

## Testing rules
- One test file per source file, mirroring directory structure
- Every module testable in isolation — no more than ~10 lines of setup per test
- @beartype active in all tests, stripped in prod
- Test shape contracts, not numeric values

## Dependencies
- PyTorch native first: F.scaled_dot_product_attention, flex_attention
- Third-party (flash-attn, local-attention) only in separate optional kernel files
- Never import third-party kernels at module load time — import inside __init__ of the kernel class

## x-transformers reference
Cloned at ./x-transformers/ — read for math and implementation details only.
Do not copy its architecture.

## Commands
```bash
# Install (uv)
uv sync --extra dev

# Lint
ruff check stackformer tests
ruff format --check stackformer tests

# Type-check
pyright stackformer

# Tests
pytest --cov=stackformer tests/
```
