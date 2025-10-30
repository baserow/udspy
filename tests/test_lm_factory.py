"""Tests for the registry-based LM factory."""

import pytest

from udspy.lm import LM
from udspy.lm.factory import PROVIDER_REGISTRY, _detect_provider
from udspy.lm.openai import OpenAILM


def test_provider_registry_structure():
    """Test that provider registry has correct structure."""
    assert "openai" in PROVIDER_REGISTRY
    assert "groq" in PROVIDER_REGISTRY
    assert "bedrock" in PROVIDER_REGISTRY
    assert "ollama" in PROVIDER_REGISTRY

    # Check that each provider has required fields
    for provider_name, config in PROVIDER_REGISTRY.items():
        assert "model_prefixes" in config
        assert "base_url_keywords" in config
        assert "default_base_url" in config
        assert "requires_api_key" in config
        assert isinstance(config["model_prefixes"], list)
        assert isinstance(config["base_url_keywords"], list)
        assert isinstance(config["requires_api_key"], bool)


def test_detect_provider_from_model_prefix():
    """Test provider detection from model prefix."""
    assert _detect_provider("groq/llama-3-70b", None) == "groq"
    assert _detect_provider("ollama/llama2", None) == "ollama"
    assert _detect_provider("bedrock/claude-3", None) == "bedrock"
    assert _detect_provider("gpt-4o", None) == "openai"  # Default


def test_detect_provider_from_base_url():
    """Test provider detection from base_url."""
    # Note: Groq URL contains "openai" but "groq" is checked first in registry
    assert _detect_provider("model", "https://api.groq.com") == "groq"
    assert _detect_provider("model", "http://localhost:11434/v1") == "ollama"
    assert _detect_provider("model", "https://bedrock.amazonaws.com") == "bedrock"
    assert _detect_provider("model", "https://api.openai.com/v1") == "openai"


def test_lm_factory_returns_openai_lm():
    """Test that LM factory returns OpenAILM instances."""
    lm = LM(model="gpt-4o", api_key="sk-test")
    assert isinstance(lm, OpenAILM)

    lm = LM(model="groq/llama-3-70b", api_key="gsk-test")
    assert isinstance(lm, OpenAILM)

    lm = LM(model="ollama/llama2")  # No API key needed
    assert isinstance(lm, OpenAILM)


def test_lm_factory_requires_api_key_for_cloud_providers():
    """Test that cloud providers require API keys."""
    # OpenAI requires API key
    with pytest.raises(ValueError, match="API key required for openai"):
        LM(model="gpt-4o")

    # Groq requires API key
    with pytest.raises(ValueError, match="API key required for groq"):
        LM(model="groq/llama-3-70b")

    # Bedrock requires API key
    with pytest.raises(ValueError, match="API key required for bedrock"):
        LM(model="bedrock/claude-3", base_url="https://bedrock.us-east-1.amazonaws.com")

    # Ollama doesn't require API key
    lm = LM(model="ollama/llama2")  # Should not raise
    assert isinstance(lm, OpenAILM)


def test_lm_factory_default_base_urls():
    """Test that providers get correct default base URLs."""
    # Groq should get default base URL
    lm = LM(model="groq/llama-3-70b", api_key="gsk-test")
    assert lm.client.base_url is not None
    assert "groq" in str(lm.client.base_url).lower()

    # Ollama should get localhost default
    lm = LM(model="ollama/llama2")
    assert lm.client.base_url is not None
    assert "11434" in str(lm.client.base_url)

    # OpenAI should use None (default)
    lm = LM(model="gpt-4o", api_key="sk-test")
    # OpenAI client doesn't expose base_url directly when using default


def test_lm_factory_custom_base_url_override():
    """Test that custom base_url overrides default."""
    custom_url = "https://custom.endpoint.com/v1"
    lm = LM(model="gpt-4o", api_key="sk-test", base_url=custom_url)

    assert lm.client.base_url is not None
    # AsyncOpenAI adds trailing slash
    assert custom_url in str(lm.client.base_url)


def test_lm_factory_cleans_model_prefix():
    """Test that model prefixes are removed."""
    lm = LM(model="groq/llama-3-70b", api_key="gsk-test")
    assert lm.default_model == "llama-3-70b"  # Prefix removed

    lm = LM(model="ollama/llama2")
    assert lm.default_model == "llama2"  # Prefix removed

    lm = LM(model="gpt-4o", api_key="sk-test")
    assert lm.default_model == "gpt-4o"  # No prefix to remove
