set dotenv-load := false

# List available recipes
default:
    @just --list

# ── Dependencies ────────────────────────────────────────────────────────────

# Install all dependencies (including dev)
sync:
    uv sync --group dev

# ── Formatting ──────────────────────────────────────────────────────────────

# Format source and tests
fmt:
    uv run ruff format stackformers/ tests/

# Check formatting without modifying files
fmt-check:
    uv run ruff format --check stackformers/ tests/

# ── Linting ─────────────────────────────────────────────────────────────────

# Lint source and tests
lint:
    uv run ruff check stackformers/ tests/

# Lint and auto-fix what's safe
lint-fix:
    uv run ruff check --fix stackformers/ tests/

# ── Type checking ────────────────────────────────────────────────────────────

# Type-check with pyrefly
types:
    uv run pyrefly check stackformers/ tests/

# ── Testing ─────────────────────────────────────────────────────────────────

# Run tests
test:
    uv run pytest tests/ -q

# Run tests with coverage report
test-cov:
    uv run pytest tests/ --cov=stackformers --cov-report=term-missing -q

# Run a single test file or pattern, e.g.: just test-only tests/v1/test_encoder.py
test-only target:
    uv run pytest {{ target }} -v

# ── Combined checks ──────────────────────────────────────────────────────────

# Run fmt-check + lint + types + test (CI gate)
check: fmt-check lint types test

# Format, lint-fix, then run all checks
fix: fmt lint-fix
    just check

# ── Housekeeping ─────────────────────────────────────────────────────────────

# Remove build artifacts and caches
clean:
    rm -rf dist/ .venv/ .ruff_cache/ .pytest_cache/ __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
