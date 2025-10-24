# udspy

A minimal DSPy-inspired library.

## Features

- **Pydantic-based Signatures**: Define inputs, outputs, and tools using Pydantic models
- **Native Tool Calling**: Uses OpenAI's native function calling API
- **Module Abstraction**: Compose LLM calls with reusable modules
- **Streaming Support**: Stream reasoning and output fields incrementally
- **Minimal Dependencies**: Only requires `openai` and `pydantic`

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
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
# Install dependencies
just install

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
