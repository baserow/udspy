"""Tests for settings and context management."""

from openai import AsyncOpenAI, OpenAI

from udspy import settings


def test_configure_with_api_key() -> None:
    """Test configuring settings with API key."""
    settings.configure(api_key="sk-test-key", model="gpt-4")

    assert settings.default_model == "gpt-4"
    assert isinstance(settings.client, OpenAI)
    assert isinstance(settings.async_client, AsyncOpenAI)


def test_configure_with_custom_client() -> None:
    """Test configuring with custom clients."""
    custom_client = OpenAI(api_key="sk-custom")
    custom_async_client = AsyncOpenAI(api_key="sk-custom")

    settings.configure(client=custom_client, async_client=custom_async_client)

    assert settings.client == custom_client
    assert settings.async_client == custom_async_client


def test_context_override_model() -> None:
    """Test context manager overrides model."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    assert settings.default_model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.default_model == "gpt-4"

    # Back to global settings
    assert settings.default_model == "gpt-4o-mini"


def test_context_override_api_key() -> None:
    """Test context manager creates new client with different API key."""
    settings.configure(api_key="sk-global")
    global_client = settings.client

    with settings.context(api_key="sk-context"):
        context_client = settings.client
        assert context_client != global_client

    # Back to global client
    assert settings.client == global_client


def test_context_override_kwargs() -> None:
    """Test context manager overrides default kwargs."""
    settings.configure(api_key="sk-test", temperature=0.5)

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

    assert settings.default_model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.default_model == "gpt-4"

        with settings.context(model="gpt-4-turbo"):
            assert settings.default_model == "gpt-4-turbo"

        # Back to outer context
        assert settings.default_model == "gpt-4"

    # Back to global
    assert settings.default_model == "gpt-4o-mini"


def test_context_with_custom_client() -> None:
    """Test context manager with custom client."""
    settings.configure(api_key="sk-global")

    custom_client = OpenAI(api_key="sk-custom")
    custom_async_client = AsyncOpenAI(api_key="sk-custom")

    with settings.context(client=custom_client, async_client=custom_async_client):
        assert settings.client == custom_client
        assert settings.async_client == custom_async_client

    # Back to global clients
    assert settings.client != custom_client
    assert settings.async_client != custom_async_client
