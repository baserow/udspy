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

# Run linter and type checker
lint:
    uv run ruff check src tests examples
    uv run mypy src

# Format code and fix linting issues
fmt:
    uv run ruff check --fix src tests examples
    uv run ruff format src tests examples

# Run type checker only
typecheck:
    uv run mypy src

# Run all checks (lint, test)
check: lint test

# Pre-release checks - run everything that CI runs
release-check:
    @echo "Running pre-release checks..."
    @echo ""
    @echo "1. Running linter and type checker..."
    just lint
    @echo ""
    @echo "2. Running tests with coverage..."
    just test
    @echo ""
    @echo "3. Building documentation..."
    just docs-build
    @echo ""
    @echo "4. Building package..."
    just build
    @echo ""
    @echo "✅ All pre-release checks passed! Ready to release."

# Build documentation
docs-build:
    uv run mkdocs build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy

# Clean build artifacts
clean:
    rm -rf dist build *.egg-info htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache

# Build package
build:
    uv build

# Run example
example name:
    uv run python examples/{{name}}.py
