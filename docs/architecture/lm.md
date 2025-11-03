# Language Model (LM) Abstraction

The LM abstraction layer provides a unified interface for interacting with different language model providers in udspy. Through a single factory function and registry-based provider detection, you can seamlessly work with OpenAI, Groq, AWS Bedrock, Ollama, and custom providers.

## Overview

The LM abstraction consists of:

1. **`LM()` factory function** - Creates provider-specific LM instances with auto-detection
2. **Provider registry** - Maps provider names to configuration (base URLs, etc.)
3. **`BaseLM` abstract class** - Interface all providers must implement
4. **`OpenAILM` implementation** - Native OpenAI support and OpenAI-compatible providers
5. **Settings integration** - Seamless configuration and context management

## Quick Start

### Basic Usage

```python
import udspy
from udspy import LM

# Configure from environment variables (UDSPY_LM_MODEL, UDSPY_LM_API_KEY)
udspy.settings.configure()

# Or use explicit LM instance
lm = LM(model="gpt-4o-mini", api_key="sk-...")
udspy.settings.configure(lm=lm)
```

### Multiple Providers

```python
# OpenAI (default)
lm = LM(model="gpt-4o", api_key="sk-...")

# Groq (via model prefix)
lm = LM(model="groq/llama-3-70b", api_key="gsk-...")

# Ollama (local, no API key needed)
lm = LM(model="ollama/llama2")

# Custom endpoint (explicit base_url)
lm = LM(
    model="llama-3-70b",
    api_key="...",
    base_url="https://api.groq.com/openai/v1"
)
```

## LM Factory Function

The `LM()` factory function provides a litellm-style interface for creating language model instances:

```python
from udspy import LM

lm = LM(
    model: str,                    # Required: model identifier
    api_key: str | None = None,    # Optional: API key (not needed for Ollama)
    base_url: str | None = None,   # Optional: custom endpoint
    **kwargs                       # Optional: client configuration
) -> BaseLM
```

### Provider Detection

The factory auto-detects the provider from:

1. **Model prefix**: `"groq/llama-3-70b"` → Groq provider
2. **Base URL keywords**: `"https://api.groq.com"` → Groq provider
3. **Fallback**: OpenAI provider

### Supported Providers

| Provider | Prefix | Implementation | API Key Required |
|----------|--------|----------------|-----------------|
| OpenAI | None (default) | Native via `openai` library | Yes |
| Groq | `groq/` | OpenAI-compatible endpoint | Yes |
| AWS Bedrock | `bedrock/` | OpenAI-compatible endpoint | Yes |
| Ollama | `ollama/` | OpenAI-compatible endpoint | No |

### Provider Examples

```python
from udspy import LM

# OpenAI
lm = LM(model="gpt-4o-mini", api_key="sk-...")

# Groq with prefix
lm = LM(model="groq/llama-3-70b", api_key="gsk-...")

# Groq without prefix (explicit base_url)
lm = LM(
    model="llama-3.1-70b-versatile",
    api_key="gsk-...",
    base_url="https://api.groq.com/openai/v1"
)

# Ollama (local)
lm = LM(model="ollama/llama2")  # No API key needed

# Ollama with explicit base_url
lm = LM(model="llama2", base_url="http://localhost:11434/v1")

# AWS Bedrock
lm = LM(
    model="bedrock/anthropic.claude-3",
    api_key="...",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
)
```

## Provider Registry

The provider registry maps provider names to default configuration and implementation classes:

```python
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": {
        "default_base_url": None,  # Uses OpenAI's default
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("UDSPY_LM_API_KEY"),
    },
    "groq": {
        "default_base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY") or os.getenv("UDSPY_LM_API_KEY"),
    },
    "bedrock": {
        "default_base_url": None,  # Region-specific, must be provided
        "api_key": os.getenv("AWS_BEDROCK_API_KEY") or os.getenv("UDSPY_LM_API_KEY"),
    },
    "ollama": {
        "default_base_url": "http://localhost:11434/v1",
        "api_key": os.getenv("UDSPY_LM_API_KEY"),
    },
}
# Note: All providers use OpenAILM implementation (OpenAI-compatible APIs)
```

### Adding Custom Providers

To add a new provider to the registry:

```python
from udspy.lm.factory import PROVIDER_REGISTRY

# Add your custom provider
PROVIDER_REGISTRY["myapi"] = {
    "default_base_url": "https://api.myservice.com/v1",
}

# Now you can use it with model prefix
from udspy import LM
lm = LM(model="myapi/my-model", api_key="...")
```

## BaseLM Abstract Class

All LM implementations must implement the `BaseLM` interface:

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseLM(ABC):
    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate a completion from the language model."""
        pass
```

**Parameters:**
- `messages`: List of messages in OpenAI format
- `model`: Optional model override (uses default if not provided)
- `tools`: Optional tool schemas in OpenAI format
- `stream`: If True, return streaming response
- `**kwargs`: Provider-specific parameters (temperature, etc.)

## OpenAILM Implementation

`OpenAILM` provides the native OpenAI implementation:

```python
from udspy.lm import OpenAILM

# Create directly
lm = OpenAILM(api_key="sk-...", default_model="gpt-4o")

# Access the model
print(lm.model)  # "gpt-4o"

# Use directly
response = await lm.acomplete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

**Key features:**
- Uses the official `openai` library
- Supports default model (optional override per call)
- Passes through all OpenAI parameters
- Handles both streaming and non-streaming
- Used for all OpenAI-compatible providers (Groq, Bedrock, Ollama, etc.)

## Settings Integration

The LM abstraction is deeply integrated with udspy's settings system.

### Configuration Methods

```python
import udspy
from udspy import LM

# Method 1: Auto-create from environment variables
# Set: UDSPY_LM_MODEL=gpt-4o, UDSPY_LM_API_KEY=sk-...
udspy.settings.configure()

# Method 2: Provide LM instance
lm = LM(model="gpt-4o", api_key="sk-...")
udspy.settings.configure(lm=lm)

# Method 3: With Groq
lm = LM(model="groq/llama-3-70b", api_key="gsk-...")
udspy.settings.configure(lm=lm)

# Method 4: With callbacks and kwargs
lm = LM(model="gpt-4o", api_key="sk-...")
udspy.settings.configure(lm=lm, callbacks=[MyCallback()], temperature=0.7)
```

### Accessing the LM

```python
# Get the configured LM
lm = udspy.settings.lm

# Access the underlying client
client = udspy.settings.lm.client

# Get the model
model = udspy.settings.lm.model

# Use directly
response = await lm.acomplete(
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Context Manager Support

Use context managers for per-request LM overrides:

```python
import udspy
from udspy import LM

# Global settings
global_lm = LM(model="gpt-4o-mini", api_key="global-key")
udspy.settings.configure(lm=global_lm)

# Temporary override with different LM
context_lm = LM(model="gpt-4", api_key="tenant-key")
with udspy.settings.context(lm=context_lm):
    result = predictor(question="...")  # Uses gpt-4 with tenant-key

# Temporary override with Groq
groq_lm = LM(model="groq/llama-3-70b", api_key="gsk-...")
with udspy.settings.context(lm=groq_lm):
    result = predictor(question="...")  # Uses Groq

# Back to global settings
result = predictor(question="...")  # Uses gpt-4o-mini with global-key
```

### Multi-Tenant Applications

Perfect for serving different users with different API keys:

```python
async def handle_user_request(user):
    # Each user can have their own LM configuration
    user_lm = LM(model=user.preferred_model, api_key=user.api_key)

    with udspy.settings.context(lm=user_lm):
        result = predictor(question=user.question)
        return result
```

## Implementing Custom Providers

### Option 1: Use Existing Registry

If your provider has an OpenAI-compatible API:

```python
from udspy import LM

# Just provide the base_url
lm = LM(
    model="my-model",
    api_key="...",
    base_url="https://api.myprovider.com/v1"
)
```

### Option 2: Extend BaseLM

For providers that need format conversion:

```python
from typing import Any
from udspy.lm import BaseLM

class AnthropicLM(BaseLM):
    """Anthropic Claude implementation."""

    def __init__(self, api_key: str, default_model: str | None = None):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    @property
    def model(self) -> str | None:
        return self._default_model

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Generate completion using Anthropic API."""
        actual_model = model or self._default_model
        if not actual_model:
            raise ValueError("No model specified")

        # Convert OpenAI format to Anthropic format
        anthropic_messages = self._convert_messages(messages)
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

    def _convert_messages(self, messages):
        """Convert OpenAI format to Anthropic format."""
        # Implementation...
        pass

    def _convert_tools(self, tools):
        """Convert OpenAI tools to Anthropic tools."""
        # Implementation...
        pass
```

### Use Custom Provider

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

## Best Practices

### For Users

1. **Use model prefixes** for clarity: `"groq/llama-3-70b"` instead of manual base_url
2. **Store API keys in environment variables** - never hardcode
3. **Use context managers** for multi-tenant scenarios
4. **Always specify a model** to avoid runtime errors
5. **Prefer `settings.lm.client`** over deprecated `settings.aclient`

### For Provider Implementers

1. **Convert to/from OpenAI format** in your implementation
2. **Handle streaming properly** - return appropriate type when `stream=True`
3. **Validate required parameters** - raise clear errors for missing config
4. **Document provider-specific kwargs** - help users understand options
5. **Test thoroughly** - ensure compatibility with udspy modules
6. **Implement `model` property** - return the default model

## Environment Variables

udspy recognizes these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `UDSPY_LM_MODEL` | Default model | `gpt-4o-mini` |
| `UDSPY_LM_API_KEY` | API key | `sk-...` |
| `UDSPY_LM_BASE_URL` | Custom base URL | `https://api.groq.com/openai/v1` |
| `OPENAI_API_KEY` | Fallback API key | `sk-...` |

```python
# Set environment variables
export UDSPY_LM_MODEL="groq/llama-3-70b"
export UDSPY_LM_API_KEY="gsk-..."

# Configure from environment
import udspy
udspy.settings.configure()  # Uses environment variables
```

## Comparison with DSPy

| Aspect | udspy | DSPy |
|--------|-------|------|
| **Factory** | `LM()` with auto-detection | Manual provider selection |
| **Interface** | `BaseLM.acomplete()` | `LM.__call__()` |
| **Async** | Async-first | Sync-first with async support |
| **Message format** | OpenAI standard | LM-specific adapters |
| **Settings** | Integrated | Separate configuration |
| **Context support** | Built-in `settings.context()` | Manual per-call |
| **Streaming** | Single method, `stream` param | Separate methods |
| **Providers** | Registry-based | Class per provider |

## Related Documentation

- [Settings and Configuration](../examples/context_settings.md)
- [Modules Architecture](modules.md)
- [Predict Module](modules/predict.md)
- [Other Providers Example](../examples/basic_usage.md)
