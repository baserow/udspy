# List available commands
default:
    @just --list

# Install dependencies and package in editable mode
install:
    uv sync --all-extras
    uv pip install -e .

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with coverage
test-cov:
    uv run pytest --cov --cov-report=html --cov-report=term

# Run linter and type checker (matches CI exactly)
lint:
    uv run ruff check src tests examples
    uv run ruff format --check src tests examples
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
    uv run mkdocs build --strict

# Serve documentation locally
docs-serve *ARGS:
    uv run mkdocs serve {{ARGS}}

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy --force

# Clean build artifacts
clean:
    rm -rf dist build *.egg-info htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache

# Build package
build:
    uv build

# Run example
example name:
    uv run python examples/{{name}}.py

# Bump version and create release branch (e.g., just bump-release 0.1.4)
bump-release version:
    @echo "🚀 Starting release process for version {{version}}..."
    @echo ""
    @echo "Validating version format..."
    @if ! echo "{{version}}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
        echo "❌ Error: Invalid version format '{{version}}'"; \
        echo "   Version must be three numbers separated by dots (e.g., 0.1.4)"; \
        exit 1; \
    fi
    @echo "✓ Version format is valid"
    @echo ""
    @./scripts/validate-version.sh {{version}}
    @echo ""
    @echo "Step 1: Running pre-release checks..."
    just release-check
    @echo ""
    @echo "Step 2: Creating release branch..."
    git checkout -b release/v{{version}}
    @echo ""
    @echo "Step 3: Updating version in pyproject.toml..."
    sed -i '' 's/^version = ".*"/version = "{{version}}"/' pyproject.toml
    @echo ""
    @echo "Step 4: Updating lockfile..."
    uv lock
    @echo ""
    @echo "Step 5: Committing changes..."
    git add pyproject.toml uv.lock
    git commit -m "chore: bump version to {{version}}"
    @echo ""
    @echo "Step 6: Pushing release branch..."
    git push -u origin release/v{{version}}
    @echo ""
    @echo "✅ Release branch created successfully!"
    @echo ""
    @echo "📋 Next steps:"
    @echo "   1. Create a PR from release/v{{version}} to main"
    @echo "   2. Get the PR reviewed and merged"
    @echo "   3. After merge, run: just create-release-tag {{version}}"
    @echo ""

# Create and push release tag (run after PR is merged to main)
create-release-tag version:
    @echo "🏷️  Creating release tag for version {{version}}..."
    @echo ""
    @echo "Validating version format..."
    @if ! echo "{{version}}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
        echo "❌ Error: Invalid version format '{{version}}'"; \
        echo "   Version must be three numbers separated by dots (e.g., 0.1.4)"; \
        exit 1; \
    fi
    @echo "✓ Version format is valid"
    @echo ""
    @echo "Step 1: Checking out main branch..."
    git checkout main
    @echo ""
    @echo "Step 2: Pulling latest changes..."
    git pull origin main
    @echo ""
    @echo "Step 3: Verifying version in pyproject.toml..."
    @if ! grep -q 'version = "{{version}}"' pyproject.toml; then \
        echo "❌ Error: Version {{version}} not found in pyproject.toml"; \
        echo "   Make sure the release PR has been merged to main"; \
        exit 1; \
    fi
    @echo "✓ Version {{version}} confirmed in pyproject.toml"
    @echo ""
    @echo "Step 4: Creating tag..."
    git tag -a "v{{version}}" -m "Release v{{version}}"
    @echo ""
    @echo "Step 5: Pushing tag..."
    git push origin "v{{version}}"
    @echo ""
    @echo "✅ Release tag v{{version}} created and pushed successfully!"
    @echo ""
    @echo "🎉 Release complete! The tag will trigger CI to build and publish."
