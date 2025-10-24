# Contributing to udspy

Thank you for your interest in contributing to udspy!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/udspy.git
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

Releases are automated through GitHub Actions. Here's how to create a new release:

### 1. Update Version

Update the version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Update this
```

And in `src/udspy/__init__.py`:

```python
__version__ = "0.2.0"  # Update this
```

### 2. Commit and Tag

```bash
# Commit version bump
git add pyproject.toml src/udspy/__init__.py
git commit -m "chore: bump version to 0.2.0"

# Create and push tag
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

### 3. Automated Release

Once the tag is pushed, GitHub Actions will automatically:
1. Run all tests
2. Build the package
3. Publish to PyPI (requires PyPI trusted publishing setup)
4. Generate changelog from commits
5. Create GitHub release with changelog
6. Comment on related issues

### 4. Verify Release

- Check [PyPI](https://pypi.org/project/udspy/) for the new version
- Check [GitHub Releases](https://github.com/your-username/udspy/releases) for the release notes
- Verify documentation is updated at the docs site

## PyPI Publishing Setup

For automated PyPI publishing to work, you need to set up Trusted Publishing:

1. Go to [PyPI Trusted Publishing](https://pypi.org/manage/account/publishing/)
2. Add a new publisher:
   - Owner: `your-username`
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

The docs will be available at: `https://your-username.github.io/udspy/`

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
