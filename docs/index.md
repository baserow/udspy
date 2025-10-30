# udspy

A minimal DSPy-inspired library with native OpenAI tool calling.

## Overview

udspy provides a clean, minimal abstraction for building LLM-powered applications with structured inputs and outputs. Inspired by DSPy, it focuses on simplicity and leverages OpenAI's native tool calling capabilities.

## Key Features

- **Pydantic-based Signatures**: Define clear input/output contracts using Pydantic models
- **Native Tool Calling**: First-class support for OpenAI's function calling API
- **Module Abstraction**: Compose LLM calls into reusable, testable modules
- **Streaming Support**: Stream reasoning and outputs incrementally for better UX
- **Minimal Dependencies**: Only requires `openai` and `pydantic`

## Quick Start

### Installation

```bash
pip install udspy
```

Or with uv:

```bash
uv add udspy
```

### Basic Usage

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict, LM

# Configure with LM instance
lm = LM(model="gpt-4o-mini", api_key="your-api-key")
udspy.settings.configure(lm=lm)

# Define a signature
class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")

# Create and use a predictor
predictor = Predict(QA)
result = predictor(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

## Philosophy

udspy is designed with these principles:

1. **Simplicity First**: Start minimal, iterate based on real needs
2. **Type Safety**: Leverage Pydantic for runtime validation
3. **Native Integration**: Use platform features (like OpenAI tools) instead of reinventing
4. **Testability**: Make it easy to test LLM-powered code
5. **Composability**: Build complex behavior from simple, reusable modules

## Comparison with DSPy

| Feature | udspy | DSPy |
|---------|-------|------|
| Input/Output Definition | Pydantic models | Custom signatures |
| Tool Calling | Native OpenAI tools | Custom adapter layer |
| Streaming | Built-in async support | Complex callback system |
| Dependencies | 2 (openai, pydantic) | Many |
| Focus | Minimal, opinionated | Full-featured framework |

## Next Steps

- Read the [Architecture Overview](architecture/overview.md)
- Check out [Examples](examples/basic_usage.md)
- Browse the [API Reference](api/signature.md)

## License

MIT License - see LICENSE file for details.
