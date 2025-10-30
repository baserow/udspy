"""Tests for settings and context management."""

from openai import AsyncOpenAI

from udspy import settings
from udspy.lm import LM, OpenAILM


def test_configure_with_lm() -> None:
    """Test configuring settings with LM."""
    lm = LM(model="gpt-4", api_key="sk-test-key")
    settings.configure(lm=lm)

    assert settings.lm.model == "gpt-4"
    assert isinstance(settings.lm.client, AsyncOpenAI)


def test_configure_with_custom_client() -> None:
    """Test configuring with custom async client."""
    custom_aclient = AsyncOpenAI(api_key="sk-custom")
    custom_lm = OpenAILM(client=custom_aclient, default_model="gpt-4o")

    settings.configure(lm=custom_lm)

    assert settings.lm.client == custom_aclient


def test_context_override_lm() -> None:
    """Test context manager overrides LM."""
    global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
    settings.configure(lm=global_lm)

    assert settings.lm.model == "gpt-4o-mini"

    context_lm = LM(model="gpt-4", api_key="sk-context")
    with settings.context(lm=context_lm):
        assert settings.lm.model == "gpt-4"

    # Back to global settings
    assert settings.lm.model == "gpt-4o-mini"


def test_context_override_api_key() -> None:
    """Test context manager creates new LM with different API key."""
    global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
    settings.configure(lm=global_lm)
    global_client = settings.lm.client

    context_lm = LM(model="gpt-4o", api_key="sk-context")
    with settings.context(lm=context_lm):
        context_client = settings.lm.client
        assert context_client != global_client

    # Back to global client
    assert settings.lm.client == global_client


def test_context_override_kwargs() -> None:
    """Test context manager overrides default kwargs."""
    lm = LM(model="gpt-4o-mini", api_key="sk-test")
    settings.configure(lm=lm, temperature=0.5)

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
    global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
    settings.configure(lm=global_lm)

    assert settings.lm.model == "gpt-4o-mini"

    lm1 = LM(model="gpt-4", api_key="sk-test")
    with settings.context(lm=lm1):
        assert settings.lm.model == "gpt-4"

        lm2 = LM(model="gpt-4-turbo", api_key="sk-test")
        with settings.context(lm=lm2):
            assert settings.lm.model == "gpt-4-turbo"

        # Back to outer context
        assert settings.lm.model == "gpt-4"

    # Back to global
    assert settings.lm.model == "gpt-4o-mini"


def test_context_with_custom_client() -> None:
    """Test context manager with custom async client."""
    global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
    settings.configure(lm=global_lm)

    custom_aclient = AsyncOpenAI(api_key="sk-custom")
    custom_lm = OpenAILM(client=custom_aclient, default_model="gpt-4o")

    with settings.context(lm=custom_lm):
        assert settings.lm.client == custom_aclient

    # Back to global clients
    assert settings.lm.client != custom_aclient


def test_context_preserves_lm_when_only_changing_other_settings() -> None:
    """Test that LM is preserved when context only changes callbacks/kwargs."""
    lm = LM(model="gpt-4o-mini", api_key="sk-global")
    settings.configure(lm=lm)
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

    # Test 3: Providing a new LM creates a new LM instance
    new_lm = LM(model="gpt-4", api_key="sk-test")
    with settings.context(lm=new_lm):
        assert settings.lm is not original_lm

    # After all contexts, should be back to original LM
    assert settings.lm is original_lm
