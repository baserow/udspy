# udspy

A minimal DSPy-inspired library.

## Features

- **Pydantic-based Signatures**: Define inputs, outputs, and tools using Pydantic models
- **Automatic Tool Calling**: Use `@tool` decorator for automatic tool execution with multi-turn conversations
- **Conversation History**: Built-in `History` class for managing multi-turn conversations
- **Optional Tool Execution**: Control whether tools execute automatically or return for manual handling
- **Module Abstraction**: Compose LLM calls with reusable modules
- **Streaming Support**: Stream reasoning and output fields incrementally with async generators
- **Minimal Dependencies**: Only requires `openai` and `pydantic`

## Installation

### For Development

```bash
# Clone the repository
git clone <your-repo-url>
cd udspy

# Install dependencies and package in editable mode
uv sync
uv pip install -e .

# Or with pip
pip install -e .
```

### For Users

```bash
# When published to PyPI
pip install udspy

# Or with uv
uv pip install udspy
```

## Quick Start

### Basic Usage

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict

# Configure OpenAI client
udspy.settings.configure(api_key="your-api-key", model="gpt-4o-mini")

# Define a signature
class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()

# Create and use a predictor
predictor = Predict(QA)
result = predictor(question="What is the capital of France?")
print(result.answer)
```

### With Conversation History

```python
from udspy import History

predictor = Predict(QA)
history = History()

# Multi-turn conversation
result = predictor(question="What is Python?", history=history)
print(result.answer)

result = predictor(question="What are its main features?", history=history)
print(result.answer)  # Context from previous turn is maintained
```

### With Automatic Tool Calling

```python
from udspy import tool
from pydantic import Field

@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

predictor = Predict(QA, tools=[calculator])
result = predictor(question="What is 157 times 234?")
print(result.answer)  # Tools are automatically executed
```

## Development

```bash
# Install dependencies and package in editable mode
just install
uv pip install -e .

# Run tests
just test

# Run linter
just lint

# Format code
just fmt

# Type check
just typecheck

# Run all checks
just check

# Build docs
just docs-serve
```

## Documentation

Full documentation is available at [your-username.github.io/udspy](https://your-username.github.io/udspy)

Or browse locally:
- [Architecture](docs/architecture/overview.md)
- [Examples](docs/examples/)
- [API Reference](docs/api/)

### Building Documentation

```bash
# Install mkdocs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Releases

Releases are automated via GitHub Actions:

1. Update version in `pyproject.toml` and `src/udspy/__init__.py`
2. Commit and tag: `git tag v0.x.x && git push --tags`
3. GitHub Actions will build, test, and publish to PyPI
4. Documentation will be deployed to GitHub Pages

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed release instructions.

## License

MIT
