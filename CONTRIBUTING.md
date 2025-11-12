# Contributing to udspy

Thank you for your interest in contributing to udspy!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/silvestrid/udspy.git
cd udspy
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync
uv pip install -e .

# Or using pip
pip install -e ".[dev]"
```

3. Install pre-commit hooks (optional but recommended):
```bash
pre-commit install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
just test

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/test_history.py -v
```

### Code Quality

```bash
# Format code
just fmt

# Run linter
just lint

# Type check
just typecheck

# Run all checks
just check
```

### Documentation

```bash
# Build and serve docs locally
just docs-serve

# Or directly with mkdocs
mkdocs serve
```

Then visit http://127.0.0.1:8000

## Pull Request Process

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Write code, tests, and documentation
3. **Run checks**: Ensure all tests pass and code is formatted
4. **Commit**: Use conventional commits (see below)
5. **Push**: `git push origin feature/your-feature-name`
6. **Open PR**: Describe your changes and link related issues

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Test additions or changes
- `chore:` - Build process or auxiliary tool changes

Examples:
```
feat: add History class for conversation management
fix: handle None values in default_model property
docs: add examples for optional tool execution
refactor: split large astream method into smaller functions
```

## Release Process

Releases are created through a branch-based workflow with automated GitHub Actions. Here's the complete process:

### 1. Create Release Branch

Use the `bump-release` just command to automate the version bump and branch creation:

```bash
just bump-release 0.2.0
```

This command will:
1. Run all pre-release checks (lint, tests, docs, build)
2. Create a new branch `release/v0.2.0`
3. Update version in `pyproject.toml`
4. Update lockfile (`uv.lock`)
5. Commit the changes
6. Push the release branch to GitHub

The command will output next steps when complete.

### 2. Create Pull Request

After the release branch is pushed:

1. Go to GitHub and create a PR from `release/v0.2.0` to `main`
2. Add title: "chore: release v0.2.0"
3. Add description summarizing the changes in this release
4. Request review if needed

### 3. Merge the PR

Once the PR is approved and CI passes:
1. Merge the PR to `main`
2. The version bump is now in the main branch

### 4. Create Release Tag

After the PR is merged, create and push the release tag:

```bash
just create-release-tag 0.2.0
```

This command will:
1. Check out the `main` branch
2. Pull the latest changes
3. Verify the version is in `pyproject.toml`
4. Create annotated tag `v0.2.0`
5. Push the tag to GitHub

### 5. Automated Release

Once the tag is pushed, GitHub Actions will automatically:
1. Run all tests
2. Build the package
3. Publish to PyPI (requires PyPI trusted publishing setup)
4. Generate changelog from commits
5. Create GitHub release with changelog
6. Comment on related issues

### 6. Verify Release

- Check [PyPI](https://pypi.org/project/udspy/) for the new version
- Check [GitHub Releases](https://github.com/silvestrid/udspy/releases) for the release notes
- Verify documentation is updated at the docs site

### Quick Reference

```bash
# Full release workflow
just bump-release 0.2.0          # Create release branch and PR
# ... merge PR on GitHub ...
just create-release-tag 0.2.0    # Create and push tag after merge

# Pre-release checks only (optional)
just release-check               # Run all CI checks locally
```

## PyPI Publishing Setup

For automated PyPI publishing to work, you need to set up Trusted Publishing:

1. Go to [PyPI Trusted Publishing](https://pypi.org/manage/account/publishing/)
2. Add a new publisher:
   - Owner: `silvestrid`
   - Repository: `udspy`
   - Workflow: `release.yml`
   - Environment: leave blank

This allows GitHub Actions to publish directly without API tokens.

## Documentation Publishing

Documentation is automatically published to GitHub Pages on every push to `main`:

1. Go to repository Settings > Pages
2. Set Source to "Deploy from a branch"
3. Select branch: `gh-pages`
4. Click Save

The docs will be available at: `https://silvestrid.github.io/udspy/`

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Keep functions short and focused (5-20 lines when possible)
- Write docstrings for all public APIs (Google style)
- Add tests for all new features and bug fixes

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage (target: >85%)
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Mock external API calls (OpenAI)
- Test both sync and async code paths

## Questions?

Feel free to open an issue for:
- Questions about contributing
- Feature requests
- Bug reports
- Documentation improvements

Thank you for contributing! 🎉
