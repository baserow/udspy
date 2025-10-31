"""Unified LM factory for creating language model instances.

This module provides a litellm-style interface where you can specify a model
and API key, and get back the appropriate LM instance for that provider.

All supported providers use OpenAI-compatible APIs.

Example:
    lm = LM(model="gpt-4o", api_key="sk-...")
    lm = LM(model="groq/llama-3-70b", api_key="gsk-...")
    lm = LM(model="ollama/llama2")
    lm = LM(model="llama-3-70b", api_key="...", base_url="https://api.groq.com/openai/v1")
"""

import os
from typing import Any, TypedDict

from openai import AsyncOpenAI

from udspy.lm.base import LM as BaseLM
from udspy.lm.openai import OpenAILM


class ProviderConfig(TypedDict):
    """Configuration for an LM provider."""

    default_base_url: str | None
    api_key: str | None


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": {
        "default_base_url": None,
        "api_key": os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    },
    "groq": {
        "default_base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("UDSPY_LM_API_KEY") or os.getenv("GROQ_API_KEY"),
    },
    "bedrock": {
        "default_base_url": None,  # Region-specific, must be provided by user
        "api_key": os.getenv("UDSPY_LM_API_KEY") or os.getenv("AWS_BEDROCK_API_KEY"),
    },
    "ollama": {
        "default_base_url": "http://localhost:11434/v1",
        "api_key": None,  # No API key needed for local Ollama
    },
}


def _detect_provider(model: str) -> str:
    """Detect provider from model string or base_url using registry.

    Detection strategy:
    - Model prefix: "groq/llama-3" → "groq"
    - Fallback: "openai"

    Args:
        model: Model identifier (e.g., "gpt-4o", "groq/llama-3", "ollama/llama2")
        base_url: Optional base URL for API

    Returns:
        Provider name from registry
    """
    if "/" in model:
        prefix = model.split("/")[0].lower()
        if prefix in PROVIDER_REGISTRY:
            return prefix

    return "openai"


def _clean_model_name(model: str) -> str:
    """Remove provider prefix from model name if present.

    Args:
        model: Model string (e.g., "groq/llama-3-70b")

    Returns:
        Clean model name (e.g., "llama-3-70b")
    """
    prefix, *rest = model.split("/", 1)
    if prefix in PROVIDER_REGISTRY and rest:
        return rest[0]
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
        lm = LM(model="gpt-4o", api_key="sk-...")

        lm = LM(model="groq/llama-3-70b", api_key="gsk-...")

        lm = LM(
            model="anthropic.claude-3",
            api_key="aws-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        )

        lm = LM(model="ollama/llama2")
        lm = LM(model="llama2", base_url="http://localhost:11434/v1")
    """
    provider = _detect_provider(model)
    config = PROVIDER_REGISTRY[provider]
    base_url = base_url or config["default_base_url"]
    provider_model = _clean_model_name(model)

    client_kwargs: dict[str, Any] = {**kwargs}
    client_kwargs["api_key"] = api_key or config.get("api_key") or "dummy"

    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)
    return OpenAILM(client, default_model=provider_model)


__all__ = ["LM"]
