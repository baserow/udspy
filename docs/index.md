# udspy

A minimal DSPy-inspired library with native OpenAI tool calling.

## Overview

udspy provides a clean, minimal abstraction for building LLM-powered applications with structured inputs and outputs. Heavily inspired by [DSPy](https://github.com/stanfordnlp/dspy), it aims to mimic DSPy's excellent API patterns while avoiding the LiteLLM dependency for resource-constrained environments.

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

## Relationship with DSPy

udspy is heavily inspired by [DSPy](https://github.com/stanfordnlp/dspy) and aims to provide a compatible API for common use cases. The main differences are:

| Aspect | udspy | DSPy |
|--------|-------|------|
| **Philosophy** | Minimal abstractions for specific use cases | Full-featured framework with optimizers |
| **Dependencies** | ~10MB (openai, pydantic) | ~200MB (includes LiteLLM, many providers) |
| **Target** | Resource-constrained environments, Baserow AI | General-purpose LLM applications |
| **Scope** | Core patterns (Predict, ChainOfThought, ReAct) | Extensive toolkit with teleprompters, optimizers |
| **Tool Calling** | OpenAI-native function calling | Provider-agnostic adapter layer |

**Use DSPy if you need**: Multiple LLM providers, optimization/teleprompters, research capabilities, full ecosystem.

**Use udspy if you need**: Minimal footprint, OpenAI-focused, simpler deployment, dynamic tool calling, reasoning and good streaming.

## Next Steps

- Read the [Architecture Overview](architecture/overview.md)
- Check out [Examples](examples/basic_usage.md)
- Browse the [API Reference](api/signature.md)

## License

MIT License - see LICENSE file for details.
