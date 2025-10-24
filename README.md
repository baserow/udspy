# udspy

A minimal DSPy-inspired library.

## Features

- **Pydantic-based Signatures**: Define inputs, outputs, and tools using Pydantic models
- **Native Tool Calling**: Uses OpenAI's native function calling API
- **Module Abstraction**: Compose LLM calls with reusable modules
- **Streaming Support**: Stream reasoning and output fields incrementally
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

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict

# Configure OpenAI client
udspy.settings.configure(api_key="your-api-key")

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

See the [documentation](docs/) for more details:

- [Architecture](docs/architecture/overview.md)
- [Examples](docs/examples/)
- [API Reference](docs/api/)

## License

MIT
