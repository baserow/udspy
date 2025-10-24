# List available commands
default:
    @just --list

# Install dependencies and package in editable mode
install:
    uv sync --all-extras
    uv pip install -e .

# Run tests
test:
    uv run pytest

# Run tests with coverage
test-cov:
    uv run pytest --cov --cov-report=html --cov-report=term

# Run linter
lint:
    uv run ruff check src tests

# Format code
fmt:
    uv run ruff format src tests

# Run type checker
typecheck:
    uv run mypy src

# Run all checks (lint, typecheck, test)
check: lint typecheck test

# Build documentation
docs-build:
    uv run mkdocs build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Clean build artifacts
clean:
    rm -rf dist build *.egg-info htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache

# Build package
build:
    uv build

# Run example
example name:
    uv run python examples/{{name}}.py
