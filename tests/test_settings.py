"""Tests for settings and context management."""

from openai import AsyncOpenAI

from udspy import settings


def test_configure_with_api_key() -> None:
    """Test configuring settings with API key."""
    settings.configure(api_key="sk-test-key", model="gpt-4")

    assert settings.lm.model == "gpt-4"
    assert isinstance(settings.lm.client, AsyncOpenAI)


def test_configure_with_custom_client() -> None:
    """Test configuring with custom async client."""
    from udspy.lm import OpenAILM

    custom_aclient = AsyncOpenAI(api_key="sk-custom")
    custom_lm = OpenAILM(client=custom_aclient, default_model="gpt-4o")

    settings.configure(lm=custom_lm)

    assert settings.lm.client == custom_aclient


def test_context_override_model() -> None:
    """Test context manager overrides model."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    assert settings.lm.model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.lm.model == "gpt-4"

    # Back to global settings
    assert settings.lm.model == "gpt-4o-mini"


def test_context_override_api_key() -> None:
    """Test context manager creates new client with different API key."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")
    global_aclient = settings.lm.client

    with settings.context(api_key="sk-context", model="gpt-4o"):
        context_aclient = settings.lm.client
        assert context_aclient != global_aclient

    # Back to global client
    assert settings.lm.client == global_aclient


def test_context_override_kwargs() -> None:
    """Test context manager overrides default kwargs."""
    settings.configure(api_key="sk-test", model="gpt-4o-mini", temperature=0.5)

    assert settings.default_kwargs["temperature"] == 0.5

    with settings.context(temperature=0.9, max_tokens=100):
        kwargs = settings.default_kwargs
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 100

    # Back to global kwargs
    assert settings.default_kwargs["temperature"] == 0.5
    assert "max_tokens" not in settings.default_kwargs


def test_nested_contexts() -> None:
    """Test nested context managers."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    assert settings.lm.model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.lm.model == "gpt-4"

        with settings.context(model="gpt-4-turbo"):
            assert settings.lm.model == "gpt-4-turbo"

        # Back to outer context
        assert settings.lm.model == "gpt-4"

    # Back to global
    assert settings.lm.model == "gpt-4o-mini"


def test_context_with_custom_client() -> None:
    """Test context manager with custom async client."""
    from udspy.lm import OpenAILM

    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    custom_aclient = AsyncOpenAI(api_key="sk-custom")
    custom_lm = OpenAILM(client=custom_aclient, default_model="gpt-4o")

    with settings.context(lm=custom_lm):
        assert settings.lm.client == custom_aclient

    # Back to global clients
    assert settings.lm.client != custom_aclient


def test_context_preserves_lm_when_only_changing_other_settings() -> None:
    """Test that LM is preserved when context only changes callbacks/kwargs."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")
    original_lm = settings.lm

    # Test 1: Only changing callbacks should keep the same LM
    from udspy import BaseCallback

    class TestCallback(BaseCallback):
        pass

    with settings.context(callbacks=[TestCallback()]):
        assert settings.lm is original_lm

    # Test 2: Only changing kwargs should keep the same LM
    with settings.context(temperature=0.9, max_tokens=100):
        assert settings.lm is original_lm
        assert settings.default_kwargs["temperature"] == 0.9

    # Test 3: Providing model/api_key/base_url creates a new LM
    with settings.context(model="gpt-4", api_key="sk-test"):
        assert settings.lm is not original_lm

    # After all contexts, should be back to original LM
    assert settings.lm is original_lm
