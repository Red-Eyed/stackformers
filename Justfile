set dotenv-load := false

python_sources := "stackformers/ tests/ examples/"

# List repository commands
default:
    @just --list

# Install all dependencies (including dev)
[group('environment')]
sync:
    uv sync --group dev

# Build distributions without retaining stale artifacts
[group('packaging')]
build:
    uv build --clear

# Format source and tests
[group('quality')]
fmt:
    uv run ruff format {{ python_sources }}

# Check formatting without modifying files
[group('quality')]
fmt-check:
    uv run ruff format --check {{ python_sources }}

# Lint source and tests
[group('quality')]
lint:
    uv run ruff check {{ python_sources }}

# Lint and auto-fix what's safe
[group('quality')]
lint-fix:
    uv run ruff check --fix {{ python_sources }}

# Type-check with pyrefly
[group('quality')]
types:
    uv run pyrefly check {{ python_sources }}

# Run tests
[group('test')]
test:
    uv run pytest tests/ -q

# Run tests with coverage report
[group('test')]
test-cov:
    uv run pytest tests/ --cov=stackformers --cov-report=term-missing -q

# Run a single test file or pattern, e.g.: just test-only tests/v1/test_encoder.py
[group('test')]
test-only target:
    uv run pytest {{ target }} -v

# Run fmt-check + lint + types + test (CI gate)
[group('quality')]
check: fmt-check lint types test

# Format, lint-fix, then run all checks
[group('quality')]
fix: fmt lint-fix
    just check

# Remove build artifacts and caches
[group('maintenance')]
clean:
    rm -rf dist/ .venv/ .ruff_cache/ .pytest_cache/ __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
