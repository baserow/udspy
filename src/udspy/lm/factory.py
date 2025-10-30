"""Unified LM factory for creating language model instances.

This module provides a litellm-style interface where you can specify a model
and API key, and get back the appropriate LM instance for that provider.

The factory uses a provider registry for easy extensibility. Each provider
is configured with:
- Model prefixes for detection (e.g., "groq/", "ollama/")
- Base URL keywords for detection (e.g., "groq", ":11434")
- Default base URL
- API key requirement

All supported providers use OpenAI-compatible APIs.

Example:
    # Auto-detect from model string
    lm = LM(model="gpt-4o", api_key="sk-...")
    lm = LM(model="groq/llama-3-70b", api_key="gsk-...")
    lm = LM(model="ollama/llama2")  # No API key needed

    # Or specify base_url
    lm = LM(model="llama-3-70b", api_key="...", base_url="https://api.groq.com/openai/v1")
"""

from typing import Any, TypedDict

from openai import AsyncOpenAI

from udspy.lm.base import LM as BaseLM
from udspy.lm.openai import OpenAILM


class ProviderConfig(TypedDict):
    """Configuration for an LM provider."""

    default_base_url: str | None  # Default base URL, or None to use provider default


# Provider registry - add new providers here
# Provider name is used for detection in both model prefix (e.g., "groq/model")
# and base_url (e.g., "https://api.groq.com")
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": {
        "default_base_url": None,  # Use OpenAI's default
    },
    "groq": {
        "default_base_url": "https://api.groq.com/openai/v1",
    },
    "bedrock": {
        "default_base_url": None,  # Must be provided by user (region-specific)
    },
    "ollama": {
        "default_base_url": "http://localhost:11434/v1",
    },
}

# Special handling for providers that don't require API keys
PROVIDERS_WITHOUT_API_KEY = {"ollama"}


def _detect_provider(model: str, base_url: str | None) -> str:
    """Detect provider from model string or base_url using registry.

    Provider detection uses the registry keys:
    - Model prefix: "groq/llama-3" → detects "groq"
    - Base URL: "https://api.groq.com" → detects "groq"
    - Special case: ":11434" → detects "ollama" (default Ollama port)

    Args:
        model: Model identifier (e.g., "gpt-4o", "groq/llama-3", "ollama/llama2")
        base_url: Optional base URL for API

    Returns:
        Provider name from registry (defaults to "openai")
    """
    # Check model prefix first
    if "/" in model:
        prefix = model.split("/")[0].lower()
        if prefix in PROVIDER_REGISTRY:
            return prefix

    # Check base_url for provider name or special markers
    if base_url:
        base_url_lower = base_url.lower()

        # Special case: Ollama's default port
        if ":11434" in base_url_lower:
            return "ollama"

        # Check if any provider name appears in base_url
        for provider_name in PROVIDER_REGISTRY:
            if provider_name in base_url_lower:
                return provider_name

    # Default to OpenAI
    return "openai"


def _clean_model_name(model: str) -> str:
    """Remove provider prefix from model name if present.

    Args:
        model: Model string (e.g., "groq/llama-3-70b")

    Returns:
        Clean model name (e.g., "llama-3-70b")
    """
    if "/" in model:
        # Remove any prefix
        return model.split("/", 1)[1]
    return model


def LM(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseLM:
    """Create a language model instance with auto-detected provider.

    This factory function detects the provider from the model string or base_url
    and returns the appropriate LM implementation. All supported providers use
    OpenAI-compatible APIs.

    Args:
        model: Model identifier. Can include provider prefix:
            - "gpt-4o" (OpenAI)
            - "groq/llama-3-70b" (Groq)
            - "bedrock/anthropic.claude-3" (AWS Bedrock)
            - "ollama/llama2" (Ollama)
        api_key: API key for the provider (not needed for Ollama)
        base_url: Optional custom base URL. Overrides provider detection.
        **kwargs: Additional parameters passed to AsyncOpenAI client

    Returns:
        LM instance configured for the detected provider

    Raises:
        ValueError: If API key is required but not provided

    Examples:
        # OpenAI
        lm = LM(model="gpt-4o", api_key="sk-...")

        # Groq
        lm = LM(model="groq/llama-3-70b", api_key="gsk-...")

        # AWS Bedrock (OpenAI-compatible endpoint)
        lm = LM(
            model="anthropic.claude-3",
            api_key="aws-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        )

        # Ollama (local, no API key)
        lm = LM(model="ollama/llama2")
        lm = LM(model="llama2", base_url="http://localhost:11434/v1")
    """
    # Detect provider using registry
    provider = _detect_provider(model, base_url)

    # Get provider configuration from registry
    config = PROVIDER_REGISTRY[provider]

    # Determine base URL (user-provided > registry default)
    actual_base_url = base_url or config["default_base_url"]

    # Clean model name (remove provider prefix if present)
    clean_model = _clean_model_name(model)

    # Validate API key requirement (all except ollama need API keys)
    if provider not in PROVIDERS_WITHOUT_API_KEY and not api_key:
        raise ValueError(
            f"API key required for {provider}. "
            f"Use provider 'ollama' for local models without API keys."
        )

    # Build AsyncOpenAI client configuration
    client_kwargs: dict[str, Any] = {**kwargs}

    # Set API key (use dummy for providers that don't need it)
    client_kwargs["api_key"] = api_key if api_key else "dummy"

    # Set base URL if needed
    if actual_base_url:
        client_kwargs["base_url"] = actual_base_url

    # Create AsyncOpenAI client (works for all OpenAI-compatible providers)
    client = AsyncOpenAI(**client_kwargs)

    # Return OpenAILM instance
    return OpenAILM(client, default_model=clean_model)


__all__ = ["LM"]
