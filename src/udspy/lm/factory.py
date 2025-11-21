"""Unified LM factory for creating language model instances.

This module provides a litellm-style interface where you can specify a model
and API key, and get back the appropriate LM instance for that provider.

All supported providers use OpenAI-compatible APIs.

Example:
    lm = LM(model="gpt-4o", api_key="sk-...")
    lm = LM(model="ollama/llama2")
    lm = LM(model="bedrock/anthropic.claude-3", api_key="...", base_url="https://...")
"""

import os
from typing import Any, TypedDict

from udspy.lm.base import LM as BaseLM
from udspy.lm.openai import OpenAILM


class ProviderConfig(TypedDict, total=False):
    """Configuration for an LM provider."""

    default_base_url: str | None
    api_key: str | None
    base_class: type[BaseLM]


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": {
        "default_base_url": None,
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
        "api_key": os.getenv("OLLAMA_API_KEY", ""),
    },
}


def _detect_provider(model: str) -> str:
    """Detect provider from model string or base_url using registry.

    Detection strategy:
    - Model prefix: "groq/llama-3" → "groq"
    - Model prefix: "ollama/llama2" → "ollama"
    - Model prefix: "bedrock/claude-3" → "bedrock"
    - Fallback: "openai"

    Args:
        model: Model identifier (e.g., "gpt-4o", "ollama/llama2", "bedrock/claude-3")

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
        model: Model string (e.g., "ollama/llama2")

    Returns:
        Clean model name (e.g., "llama2")
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

        lm = LM(
            model="bedrock/anthropic.claude-3",
            api_key="aws-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        )

        lm = LM(model="ollama/llama2")
        lm = LM(model="llama2", base_url="http://localhost:11434/v1")
    """
    provider = _detect_provider(model)
    config = PROVIDER_REGISTRY[provider]

    base_url = (
        base_url or os.getenv("UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL") or config["default_base_url"]
    )
    api_key = api_key or os.getenv("UDSPY_LM_API_KEY") or config["api_key"]
    client_kwargs: dict[str, Any] = {
        **kwargs,
        "base_url": base_url,
        "api_key": api_key,
    }

    return OpenAILM(**client_kwargs, default_model=_clean_model_name(model), provider=provider)


__all__ = ["LM"]
