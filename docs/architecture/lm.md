# Language Model (LM) Abstraction

The LM abstraction layer provides a provider-agnostic interface for interacting with language models in udspy. This allows the library to work with different LLM providers (OpenAI, Anthropic, local models, etc.) through a common interface.

## Overview

The LM abstraction consists of:

1. **`LM` base class** - Abstract interface defining the contract all providers must implement
2. **`OpenAILM` implementation** - Concrete implementation for OpenAI's API
3. **Settings integration** - Seamless integration with udspy's configuration system
4. **Context support** - Per-context provider selection for multi-tenant applications

## Core Concepts

### LM Base Class

The `LM` abstract base class defines a single method that all providers must implement:

```python
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

class LM(ABC):
    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        """Generate a completion from the language model."""
        pass
```

**Parameters:**
- `messages`: List of messages in OpenAI format: `[{"role": "user", "content": "..."}]`
- `model`: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
- `tools`: Optional tool schemas in OpenAI format
- `stream`: If True, return an async generator of chunks
- `**kwargs`: Provider-specific parameters (temperature, max_tokens, etc.)

**Returns:**
- If `stream=False`: Completion response object
- If `stream=True`: AsyncGenerator yielding completion chunks

### OpenAI Implementation

`OpenAILM` wraps the AsyncOpenAI client to provide the LM interface:

```python
from openai import AsyncOpenAI
from udspy.lm import OpenAILM

# Create OpenAI LM with default model
client = AsyncOpenAI(api_key="sk-...")
lm = OpenAILM(client, default_model="gpt-4o")

# Use directly
response = await lm.acomplete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

**Key features:**
- Wraps AsyncOpenAI client
- Supports default model (can be overridden per call)
- Passes through all OpenAI parameters
- Handles both streaming and non-streaming responses

## Settings Integration

The LM abstraction is tightly integrated with udspy's settings system:

### Configuration

Configure the LM in three ways:

```python
import udspy
from openai import AsyncOpenAI
from udspy.lm import OpenAILM

# 1. Simple: API key (creates OpenAILM automatically)
udspy.settings.configure(api_key="sk-...", model="gpt-4o")

# 2. With custom client (creates OpenAILM automatically)
client = AsyncOpenAI(api_key="sk-...", timeout=30.0)
udspy.settings.configure(aclient=client, model="gpt-4o")

# 3. With custom LM instance (full control)
lm = OpenAILM(client, default_model="gpt-4o")
udspy.settings.configure(lm=lm)
```

### Accessing the LM

Access the configured LM via `settings.lm`:

```python
# Get the configured LM
lm = udspy.settings.lm

# Use in your code
response = await lm.acomplete(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",  # Optional: override default
    temperature=0.7
)
```

**Note:** `settings.aclient` still works for backward compatibility but is deprecated. Use `settings.lm` instead.

## Context Manager Support

The LM abstraction supports context-specific overrides, useful for multi-tenant applications:

```python
import udspy
from udspy.lm import OpenAILM
from openai import AsyncOpenAI

# Global settings
udspy.settings.configure(api_key="global-key", model="gpt-4o-mini")

# Temporary override with custom LM
custom_client = AsyncOpenAI(api_key="tenant-key")
custom_lm = OpenAILM(custom_client, default_model="gpt-4o")

with udspy.settings.context(lm=custom_lm):
    # Uses custom_lm
    result = predictor(question="...")

# Back to global LM
result = predictor(question="...")  # Uses global LM
```

### Context Priority

When using the context manager:

1. **Explicit `lm` parameter** - Highest priority
2. **`aclient` parameter** - Creates OpenAILM wrapper
3. **`api_key` parameter** - Creates new client and OpenAILM wrapper
4. **Global settings** - Fallback

```python
# Priority example
with udspy.settings.context(
    lm=custom_lm,        # This takes priority
    aclient=other_client  # Ignored because lm is provided
):
    pass
```

## Implementing Custom Providers

To add support for a new LLM provider, implement the `LM` interface:

```python
from typing import Any, AsyncGenerator
from udspy.lm import LM

class AnthropicLM(LM):
    """Anthropic Claude implementation."""

    def __init__(self, api_key: str, default_model: str | None = None):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        """Generate completion using Anthropic API."""
        actual_model = model or self.default_model
        if not actual_model:
            raise ValueError("No model specified")

        # Convert OpenAI format to Anthropic format
        anthropic_messages = self._convert_messages(messages)

        # Convert OpenAI tools to Anthropic tools if needed
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Call Anthropic API
        response = await self.client.messages.create(
            model=actual_model,
            messages=anthropic_messages,
            tools=anthropic_tools,
            stream=stream,
            **kwargs
        )

        return response

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI message format to Anthropic format."""
        # Implementation details...
        pass

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        # Implementation details...
        pass
```

### Use Your Custom Provider

```python
import udspy
from my_providers import AnthropicLM

# Configure with custom provider
lm = AnthropicLM(api_key="sk-ant-...", default_model="claude-3-5-sonnet-20241022")
udspy.settings.configure(lm=lm)

# Use normally - all udspy features work!
from udspy import Predict, Signature, InputField, OutputField

class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)
result = predictor(question="What is the capital of France?")
print(result.answer)  # Uses Anthropic Claude
```

## Message Format Standard

The LM abstraction uses **OpenAI's message format** as the standard:

```python
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me a joke."}
]
```

**Why OpenAI format?**
- Industry standard - widely adopted
- Simple and flexible
- Easy to convert to other formats
- Well-documented

**Custom providers** should convert to/from OpenAI format internally.

## Tool Format Standard

Tools use OpenAI's function calling schema:

```python
[
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

Custom providers should convert tool schemas to their native format.

## Error Handling

LM implementations should handle provider-specific errors and optionally convert them to common exceptions:

```python
from openai import APIError, RateLimitError

class OpenAILM(LM):
    async def acomplete(self, ...):
        try:
            response = await self.client.chat.completions.create(...)
            return response
        except RateLimitError as e:
            # Could convert to common LMRateLimitError
            raise
        except APIError as e:
            # Could convert to common LMError
            raise
```

**Note:** Currently, udspy doesn't define common error classes, so providers can raise their native exceptions.

## Type Handling

The LM abstraction uses union return types to support both streaming and non-streaming:

```python
async def acomplete(...) -> Any | AsyncGenerator[Any, None]:
    pass
```

In Predict module, this is handled with type: ignore comments where mypy can't narrow the type:

```python
response = await settings.lm.acomplete(**kwargs)  # Could be streaming or not
message = response.choices[0].message  # type: ignore[union-attr]
```

This is acceptable because:
1. The `stream` parameter determines the actual type
2. Runtime errors are unlikely (tests verify correctness)
3. Alternative (overloads) doesn't work well with abstract base classes

## Best Practices

### For Users

1. **Use `settings.lm`** instead of `settings.aclient` for new code
2. **Prefer `configure(lm=...)`** for custom providers
3. **Use context manager** for multi-tenant scenarios
4. **Always specify a default model** to avoid runtime errors

### For Provider Implementers

1. **Convert to/from OpenAI format** in your implementation
2. **Handle streaming properly** - return AsyncGenerator when `stream=True`
3. **Validate required parameters** - raise clear errors for missing config
4. **Document provider-specific kwargs** - help users understand options
5. **Test thoroughly** - ensure compatibility with udspy modules

## Comparison with DSPy

| Aspect | udspy | DSPy |
|--------|-------|------|
| **Interface** | `LM.acomplete()` | `LM.__call__()` |
| **Async** | Async-first | Sync-first with async support |
| **Message format** | OpenAI standard | LM-specific adapters |
| **Settings** | Integrated with settings | Separate configuration |
| **Context support** | Built-in via `settings.context()` | Manual per-call |
| **Streaming** | Single method, `stream` param | Separate methods |

## Internal Usage

Within udspy, the LM is accessed via settings in the Predict module:

```python
# In Predict.aexecute() - non-streaming
response = await settings.lm.acomplete(
    messages=messages,
    model=model or settings.default_model,
    tools=tool_schemas,
    stream=False,
    **kwargs
)

# In Predict._aexecute_stream() - streaming
stream = await settings.lm.acomplete(
    messages=messages,
    model=model or settings.default_model,
    tools=tool_schemas,
    stream=True,
    **kwargs
)
```

This keeps all LLM calls centralized and makes it easy to swap providers.

## Future Enhancements

Possible future improvements:

1. **Common error classes** - `LMError`, `LMRateLimitError`, etc.
2. **Response normalization** - Standard response format across providers
3. **Built-in retry logic** - Automatic retries with exponential backoff
4. **Token counting** - Provider-agnostic token usage tracking
5. **Caching layer** - Optional caching for repeated calls
6. **Provider registry** - `settings.configure(provider="anthropic", ...)`
7. **Local model support** - Ollama, LM Studio, vLLM integration

## Related Documentation

- [Settings and Configuration](../examples/context_settings.md)
- [Modules Architecture](modules.md)
- [Predict Module](modules/predict.md)
- [Adapters](adapters.md)
