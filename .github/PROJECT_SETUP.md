# GitHub Project Setup Guide

This guide walks through the complete setup of the udspy GitHub repository.

## 1. Repository Topics/Tags

Add these topics to help users discover the project:

### How to Add Topics
1. Go to your repository main page
2. Click the gear icon ⚙️ next to "About" (right sidebar)
3. Add the following topics in the "Topics" field
4. Click "Save changes"

### Recommended Topics
```
python
openai
llm
dspy
pydantic
async
ai-framework
function-calling
tool-calling
streaming
conversational-ai
prompt-engineering
type-hints
pytest
chatbot
agent
```

## 2. Repository Settings

### About Section
- **Description**: `A minimal DSPy-inspired library with native OpenAI tool calling, conversation history, and streaming support`
- **Website**: `https://silvestrid.github.io/udspy` (update after GitHub Pages is enabled)
- **Topics**: (add the tags listed above)

### Features to Enable
- ☑️ Issues
- ☑️ Discussions (optional, for community Q&A)
- ☑️ Projects (optional, for roadmap tracking)
- ☑️ Wiki (optional, for additional documentation)

## 3. GitHub Pages Setup

Enable GitHub Pages for automatic documentation deployment:

1. Go to **Settings** > **Pages**
2. Under "Source", select **Deploy from a branch**
3. Select branch: **`gh-pages`**
4. Select folder: **`/ (root)`**
5. Click **Save**

After the first docs workflow runs, your documentation will be available at:
`https://silvestrid.github.io/udspy`

### Update Documentation URL
After GitHub Pages is enabled, update these files with your actual URL:

- `mkdocs.yml` - line 3: `site_url`
- `mkdocs.yml` - line 4-5: `repo_url` and `repo_name`
- `README.md` - documentation link
- `.github/PROJECT_SETUP.md` - this file

## 4. PyPI Trusted Publishing Setup

Enable automated PyPI publishing without API tokens:

### First-Time Setup
1. Create an account on [PyPI](https://pypi.org) if you don't have one
2. **Important**: You must create the project on PyPI first by doing a manual upload:
   ```bash
   # Build the package
   python -m build

   # Upload manually (first time only)
   python -m twine upload dist/*
   ```

### Enable Trusted Publishing
1. Go to [PyPI Trusted Publishing](https://pypi.org/manage/account/publishing/)
2. Add a new publisher with these settings:
   - **PyPI Project Name**: `udspy`
   - **Owner**: `silvestrid`
   - **Repository name**: `udspy`
   - **Workflow name**: `release.yml`
   - **Environment name**: (leave blank)
3. Click **Add**

Now GitHub Actions can publish releases automatically when you push a tag!

## 5. Codecov Integration (Optional)

Add test coverage reporting:

1. Sign up at [codecov.io](https://codecov.io) with your GitHub account
2. Add the `udspy` repository
3. Get your Codecov token
4. Add it as a repository secret:
   - Go to **Settings** > **Secrets and variables** > **Actions**
   - Click **New repository secret**
   - Name: `CODECOV_TOKEN`
   - Value: (paste your token)
   - Click **Add secret**

## 6. Branch Protection Rules (Recommended)

Protect the `main` branch:

1. Go to **Settings** > **Branches**
2. Click **Add branch protection rule**
3. Branch name pattern: `main`
4. Enable these settings:
   - ☑️ Require a pull request before merging
     - ☑️ Require approvals (1)
     - ☑️ Dismiss stale pull request approvals when new commits are pushed
   - ☑️ Require status checks to pass before merging
     - Add required checks: `test`, `lint`
   - ☑️ Require conversation resolution before merging
   - ☑️ Do not allow bypassing the above settings
5. Click **Create**

## 7. Issue Templates (Optional)

Create issue templates for better bug reports and feature requests:

### Bug Report Template
Create `.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1.
2.
3.

**Expected behavior**
What you expected to happen.

**Code example**
```python
# Minimal code to reproduce
```

**Environment**
- OS: [e.g., macOS 14, Ubuntu 22.04]
- Python version: [e.g., 3.11]
- udspy version: [e.g., 0.1.0]

**Additional context**
Any other information about the problem.
```

### Feature Request Template
Create `.github/ISSUE_TEMPLATE/feature_request.md`:
```markdown
---
name: Feature Request
about: Suggest a new feature or enhancement
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem?**
A clear description of the problem. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples.
```

## 8. Repository Labels

Add these labels for better issue organization:

### Type Labels
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `question` - Further information is requested

### Priority Labels
- `priority: high` - High priority
- `priority: medium` - Medium priority
- `priority: low` - Low priority

### Status Labels
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `wontfix` - This will not be worked on
- `duplicate` - This issue already exists

## 9. Creating Your First Release

Once everything is set up, create your first release:

1. **Update version** in `pyproject.toml` and `src/udspy/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Commit changes**:
   ```bash
   git add pyproject.toml src/udspy/__init__.py CHANGELOG.md
   git commit -m "chore: bump version to 0.1.0"
   ```
4. **Create and push tag**:
   ```bash
   git tag v0.1.0
   git push origin main
   git push origin v0.1.0
   ```
5. **Wait for automation**: GitHub Actions will build, test, and publish automatically

## 10. Post-Setup Checklist

- [ ] Repository topics added
- [ ] About section filled with description and website
- [ ] GitHub Pages enabled and documentation URL updated
- [ ] PyPI project created and trusted publishing configured
- [ ] Codecov integration added (optional)
- [ ] Branch protection rules enabled
- [ ] Issue templates created
- [ ] Repository labels organized
- [ ] First release published
- [ ] Documentation live and accessible
- [ ] Package available on PyPI

## 11. Maintenance

### Regular Tasks
- Monitor GitHub Actions for workflow failures
- Review and merge dependabot PRs (if enabled)
- Update CHANGELOG.md for each release
- Keep documentation in sync with code changes
- Respond to issues and PRs promptly

### Version Bumping
Follow [Semantic Versioning](https://semver.org/):
- **Patch** (0.1.X): Bug fixes
- **Minor** (0.X.0): New features (backward compatible)
- **Major** (X.0.0): Breaking changes

## Need Help?

- Check [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Open an issue for questions or problems
- Review GitHub Actions logs for CI/CD issues

---

Last updated: 2025-10-24
