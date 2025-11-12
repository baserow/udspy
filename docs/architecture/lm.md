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

# Configure from environment variables
# Set: UDSPY_LM_MODEL="gpt-4o-mini" and OPENAI_API_KEY="sk-..."
udspy.settings.configure()

# Or use explicit LM instance
from udspy import LM
lm = LM(model="gpt-4o-mini", api_key="sk-...")
udspy.settings.configure(lm=lm)
```

### Multiple Providers

The recommended way to work with multiple providers is to set provider-specific API keys and switch via `UDSPY_LM_MODEL`:

```bash
# .env file
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk-..."
UDSPY_LM_MODEL="gpt-4o-mini"  # Change this to switch providers
```

```python
import udspy

# Auto-configures based on UDSPY_LM_MODEL prefix
udspy.settings.configure()

# Or create LM instances directly:
from udspy import LM

# OpenAI (uses OPENAI_API_KEY from environment)
lm = LM(model="gpt-4o-mini")

# Groq (uses GROQ_API_KEY from environment)
lm = LM(model="groq/llama-3-70b")

# Ollama (local, minimal config needed)
lm = LM(model="ollama/llama2")

# Custom endpoint (only when needed)
lm = LM(
    model="my-model",
    api_key="...",
    base_url="https://my-endpoint.com/v1"
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

## Provider Configuration

udspy supports multiple LLM providers through a unified interface. All providers use OpenAI-compatible APIs, making it easy to switch between them.

### Environment Variable Precedence

When configuring providers, udspy follows this precedence order for API keys and base URLs:

**API Key Precedence** (highest to lowest):
1. Explicitly passed `api_key` parameter to `LM()`
2. `UDSPY_LM_API_KEY` environment variable (general override)
3. Provider-specific environment variable (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`)

**Base URL Precedence** (highest to lowest):
1. Explicitly passed `base_url` parameter to `LM()`
2. `UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL` environment variable (general override)
3. Provider's default base URL from registry

This precedence system allows you to:
- **Override all providers** with `UDSPY_LM_API_KEY` when you want to use the same key everywhere
- **Fall back to provider-specific keys** when `UDSPY_LM_API_KEY` is not set
- Set custom **base URLs** for self-hosted or custom endpoints

**Important**: `UDSPY_LM_API_KEY` takes **precedence** over provider-specific keys. If you want to use different keys for different providers, **don't set** `UDSPY_LM_API_KEY`.

### Best Practice: Switching Between Providers

There are two recommended approaches depending on your use case:

#### Approach 1: Provider-Specific Keys (Recommended for Multi-Provider)

Best when you need different API keys for different providers:

1. **Set provider-specific API keys** (don't set `UDSPY_LM_API_KEY`):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export GROQ_API_KEY="gsk-..."
   export AWS_BEARER_TOKEN_BEDROCK="..."
   # Don't set UDSPY_LM_API_KEY - it would override all provider-specific keys!
   ```

2. **Switch providers by changing only the model**:
   ```bash
   # Switch to OpenAI (uses OPENAI_API_KEY)
   export UDSPY_LM_MODEL="gpt-4o-mini"

   # Switch to Groq (uses GROQ_API_KEY)
   export UDSPY_LM_MODEL="groq/llama-3.1-70b-versatile"

   # Switch to Ollama (local)
   export UDSPY_LM_MODEL="ollama/llama2"
   ```

**Example**:
```bash
# .env file
OPENAI_API_KEY="sk-proj-..."
GROQ_API_KEY="gsk_..."
AWS_BEARER_TOKEN_BEDROCK="eyJ..."
AWS_REGION_NAME="us-east-1"

# Switch providers by changing this single variable:
UDSPY_LM_MODEL="gpt-4o-mini"          # Uses OPENAI_API_KEY
# UDSPY_LM_MODEL="groq/llama-3-70b"   # Uses GROQ_API_KEY
# UDSPY_LM_MODEL="bedrock/claude-3"   # Uses AWS_BEARER_TOKEN_BEDROCK
```

```python
import udspy
udspy.settings.configure()  # Auto-configures based on UDSPY_LM_MODEL
```

#### Approach 2: Single Key Override (Development/Testing)

Best when using the same API key across all providers (e.g., testing with one account):

```bash
# .env file
UDSPY_LM_API_KEY="sk-..."  # This overrides ALL provider-specific keys
UDSPY_LM_MODEL="gpt-4o-mini"
```

```python
import udspy
udspy.settings.configure()
```

**Note**: Because `UDSPY_LM_API_KEY` has higher precedence than provider-specific keys, it will be used for all providers regardless of whether `OPENAI_API_KEY`, `GROQ_API_KEY`, etc. are set.

### Provider Examples

```python
from udspy import LM

# OpenAI (uses OPENAI_API_KEY from environment)
lm = LM(model="gpt-4o-mini")

# OpenAI with explicit key
lm = LM(model="gpt-4o-mini", api_key="sk-...")

# Groq with prefix (uses GROQ_API_KEY from environment)
lm = LM(model="groq/llama-3-70b")

# Groq with explicit key
lm = LM(model="groq/llama-3-70b", api_key="gsk-...")

# Groq without prefix (explicit base_url)
lm = LM(
    model="llama-3.1-70b-versatile",
    api_key="gsk-...",
    base_url="https://api.groq.com/openai/v1"
)

# Ollama (local, uses UDSPY_LM_API_KEY or empty string)
lm = LM(model="ollama/llama2")

# Ollama with explicit base_url
lm = LM(model="llama2", base_url="http://localhost:11434/v1")

# AWS Bedrock (uses AWS_BEARER_TOKEN_BEDROCK and AWS_REGION_NAME)
lm = LM(model="bedrock/anthropic.claude-3")

# AWS Bedrock with explicit configuration
lm = LM(
    model="bedrock/anthropic.claude-3",
    api_key="eyJ...",
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
)
```

### When to Use Custom Base URLs

Only set `UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL` or pass `base_url` when:

1. **Self-hosted models**: Running your own OpenAI-compatible server
   ```bash
   export UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
   export UDSPY_LM_MODEL="my-custom-model"
   ```

2. **Proxy/Gateway**: Using a proxy that forwards to multiple providers
   ```bash
   export UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL="https://my-proxy.com/v1"
   ```

3. **Custom Ollama port**: Running Ollama on a non-standard port
   ```bash
   export UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL="http://localhost:8080/v1"
   export UDSPY_LM_MODEL="ollama/llama2"
   ```

**Don't use custom base URLs** when switching between standard providers - the registry already knows the correct endpoints!

## Provider Registry

The provider registry maps provider names to default configuration and implementation classes:

```python
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": {
        "default_base_url": None,  # Uses OpenAI's default
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "groq": {
        "default_base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY"),
    },
    "bedrock": {
        "default_base_url": f"https://bedrock-runtime.{os.getenv('AWS_REGION_NAME', 'us-east-1')}.amazonaws.com/openai/v1",
        "api_key": os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    },
    "ollama": {
        "default_base_url": "http://localhost:11434/v1",
        "api_key": os.getenv("OLLAMA_API_KEY"),
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

1. **Use provider-specific API keys** for multi-provider setups - See [Best Practice: Switching Between Providers](#best-practice-switching-between-providers)
2. **Switch providers via `UDSPY_LM_MODEL` only** - avoid changing `base_url` unless needed
3. **Use model prefixes** for clarity: `"groq/llama-3-70b"` instead of manual base_url
4. **Store API keys in environment variables** - never hardcode
5. **Use context managers** for multi-tenant scenarios
6. **Always specify a model** to avoid runtime errors
7. **Prefer `settings.lm.client`** over deprecated `settings.aclient`

### For Provider Implementers

1. **Convert to/from OpenAI format** in your implementation
2. **Handle streaming properly** - return appropriate type when `stream=True`
3. **Validate required parameters** - raise clear errors for missing config
4. **Document provider-specific kwargs** - help users understand options
5. **Test thoroughly** - ensure compatibility with udspy modules
6. **Implement `model` property** - return the default model

## Environment Variables

udspy recognizes these environment variables. See [Environment Variable Precedence](#environment-variable-precedence) for how these variables are resolved.

### General Variables

| Variable | Description | Example | Precedence |
|----------|-------------|---------|------------|
| `UDSPY_LM_MODEL` | Default model identifier | `gpt-4o-mini`, `groq/llama-3-70b` | Required for auto-config |
| `UDSPY_LM_API_KEY` | General API key override | `sk-...` | 2nd (overrides provider-specific keys) |
| `UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL` | Custom base URL override | `https://my-proxy.com/v1` | 2nd (overrides provider defaults) |

### Provider-Specific Variables

| Variable | Provider | Description | Precedence |
|----------|----------|-------------|------------|
| `OPENAI_API_KEY` | OpenAI | OpenAI API key | 3rd (fallback if no UDSPY_LM_API_KEY) |
| `GROQ_API_KEY` | Groq | Groq API key | 3rd (fallback if no UDSPY_LM_API_KEY) |
| `AWS_BEARER_TOKEN_BEDROCK` | AWS Bedrock | AWS Bedrock bearer token | 3rd (fallback if no UDSPY_LM_API_KEY) |
| `AWS_REGION_NAME` | AWS Bedrock | AWS region for Bedrock endpoint | Used for default base URL |
| `OLLAMA_API_KEY` | Ollama | Ollama API key (rarely needed) | 3rd (fallback if no UDSPY_LM_API_KEY) |

### Variable Resolution Order

**For API Keys**:
1. Explicit `api_key` parameter to `LM()`
2. `UDSPY_LM_API_KEY` (general override - **takes precedence over provider-specific keys**)
3. Provider-specific key (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`)

**For Base URLs**:
1. Explicit `base_url` parameter to `LM()`
2. `UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL` (general override - **takes precedence over provider defaults**)
3. Provider's default from registry

**Important**: To use different API keys for different providers, **do not set** `UDSPY_LM_API_KEY`. Only set provider-specific keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, etc.).

### Examples

**Example 1: Single API Key for All Providers**:
```bash
# UDSPY_LM_API_KEY overrides provider-specific keys
export UDSPY_LM_MODEL="gpt-4o-mini"
export UDSPY_LM_API_KEY="sk-..."  # Used for ALL providers
```

```python
import udspy
udspy.settings.configure()  # Uses UDSPY_LM_API_KEY for all models
```

**Example 2: Multi-Provider Setup (Recommended)**:
```bash
# Set provider-specific keys, DON'T set UDSPY_LM_API_KEY
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk-..."
# No UDSPY_LM_API_KEY - this allows provider-specific keys to work

# Switch providers by changing model only
export UDSPY_LM_MODEL="gpt-4o-mini"          # Uses OPENAI_API_KEY
# export UDSPY_LM_MODEL="groq/llama-3-70b"   # Uses GROQ_API_KEY
```

```python
import udspy
udspy.settings.configure()  # Auto-selects key based on provider
```

**Example 3: Custom Endpoint**:
```bash
# Override base URL for all providers
export UDSPY_LM_MODEL="my-model"
export UDSPY_LM_API_KEY="custom-key"
export UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
```

```python
import udspy
udspy.settings.configure()  # Uses custom endpoint
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
