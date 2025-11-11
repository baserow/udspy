"""Tests for the registry-based LM factory."""

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
    for _provider_name, config in PROVIDER_REGISTRY.items():
        assert len(config) == 2
        assert "default_base_url" in config
        assert "api_key" in config


def test_detect_provider_from_model_prefix():
    """Test provider detection from model prefix."""
    assert _detect_provider("groq/llama-3-70b") == "groq"
    assert _detect_provider("ollama/llama2") == "ollama"
    assert _detect_provider("bedrock/claude-3") == "bedrock"
    assert _detect_provider("gpt-4o") == "openai"  # Default


def test_lm_factory_default_base_urls():
    """Test that providers get correct default base URLs."""
    # Groq uses OpenAI-compatible API with custom base URL
    lm = LM(model="groq/llama-3-70b", api_key="gsk-test")
    assert isinstance(lm, OpenAILM)
    # Groq uses api.groq.com base URL
    assert lm.client.base_url is not None
    assert "groq" in str(lm.client.base_url)

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
    # Groq uses OpenAI-compatible API
    lm = LM(model="groq/llama-3-70b", api_key="gsk-test")
    assert isinstance(lm, OpenAILM)
    assert lm.default_model == "llama-3-70b"  # Prefix removed

    lm = LM(model="ollama/llama2")
    assert isinstance(lm, OpenAILM)
    assert lm.default_model == "llama2"  # Prefix removed

    lm = LM(model="gpt-4o", api_key="sk-test")
    assert isinstance(lm, OpenAILM)
    assert lm.default_model == "gpt-4o"  # No prefix to remove
