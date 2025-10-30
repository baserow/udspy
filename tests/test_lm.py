"""Tests for LM abstraction layer."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from udspy import settings
from udspy.lm import LM, BaseLM, OpenAILM


class TestOpenAILM:
    """Tests for OpenAILM implementation."""

    @pytest.mark.asyncio
    async def test_acomplete_basic(self) -> None:
        """Test basic completion call."""
        # Create mock client
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.choices = [
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create LM instance
        lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        # Call acomplete
        messages = [{"role": "user", "content": "Hi"}]
        response = await lm.acomplete(messages)

        # Verify
        assert response == mock_response
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o", messages=messages, stream=False
        )

    @pytest.mark.asyncio
    async def test_acomplete_with_model_override(self) -> None:
        """Test that explicit model parameter overrides default."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock(spec=ChatCompletion)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        messages = [{"role": "user", "content": "Hi"}]
        await lm.acomplete(messages, model="gpt-4-turbo")

        # Should use explicit model, not default
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4-turbo", messages=messages, stream=False
        )

    @pytest.mark.asyncio
    async def test_acomplete_no_model_raises_error(self) -> None:
        """Test that missing model raises ValueError."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        lm = OpenAILM(client=mock_client)  # No default_model

        messages = [{"role": "user", "content": "Hi"}]

        with pytest.raises(ValueError, match="No model specified"):
            await lm.acomplete(messages)

    @pytest.mark.asyncio
    async def test_acomplete_with_tools(self) -> None:
        """Test completion with tool schemas."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock(spec=ChatCompletion)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        messages = [{"role": "user", "content": "Calculate 2+2"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            }
        ]

        await lm.acomplete(messages, tools=tools)

        # Verify tools were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_acomplete_with_stream(self) -> None:
        """Test streaming completion."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_stream = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        messages = [{"role": "user", "content": "Hi"}]
        response = await lm.acomplete(messages, stream=True)

        # Verify stream parameter was passed
        assert response == mock_stream
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_acomplete_with_extra_kwargs(self) -> None:
        """Test that extra kwargs are passed through."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = MagicMock(spec=ChatCompletion)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        messages = [{"role": "user", "content": "Hi"}]
        await lm.acomplete(
            messages, temperature=0.7, max_tokens=100, top_p=0.9, custom_param="value"
        )

        # Verify all kwargs were passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["custom_param"] == "value"


class TestSettingsLMIntegration:
    """Tests for settings.lm integration."""

    def test_configure_creates_lm(self) -> None:
        """Test that configure() creates an LM instance."""
        lm = LM(model="gpt-4o", api_key="sk-test")
        settings.configure(lm=lm)

        assert settings.lm == lm
        assert isinstance(settings.lm, OpenAILM)

    def test_configure_with_custom_lm(self) -> None:
        """Test configuring with custom LM instance."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        custom_lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        settings.configure(lm=custom_lm)

        assert settings.lm == custom_lm

    def test_lm_not_configured_raises_error(self) -> None:
        """Test that accessing lm before configuration raises error."""
        # Reset settings
        settings._lm = None
        settings._context_lm.set(None)

        with pytest.raises(RuntimeError, match="LM not configured"):
            _ = settings.lm

        # Restore settings
        lm = LM(model="gpt-4o-mini", api_key="sk-test")
        settings.configure(lm=lm)

    def test_backward_compatibility_aclient_still_works(self) -> None:
        """Test that settings.lm.client still works (backward compatibility)."""
        lm = LM(model="gpt-4o-mini", api_key="sk-test")
        settings.configure(lm=lm)

        # Should not raise
        aclient = settings.lm.client
        assert isinstance(aclient, AsyncOpenAI)


class TestLMContextManager:
    """Tests for LM context manager."""

    def test_context_with_custom_lm(self) -> None:
        """Test context manager with custom LM instance."""
        global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
        settings.configure(lm=global_lm)

        mock_client = AsyncMock(spec=AsyncOpenAI)
        custom_lm = OpenAILM(client=mock_client, default_model="gpt-4o")

        with settings.context(lm=custom_lm):
            assert settings.lm == custom_lm
            assert settings.lm != global_lm

        # Back to global LM
        assert settings.lm == global_lm

    def test_context_with_different_lm(self) -> None:
        """Test that context with different LM creates new LM."""
        global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
        settings.configure(lm=global_lm)

        context_lm = LM(model="gpt-4o", api_key="sk-context")
        with settings.context(lm=context_lm):
            assert settings.lm == context_lm
            assert isinstance(settings.lm, OpenAILM)
            assert settings.lm != global_lm

        # Back to global LM
        assert settings.lm == global_lm

    def test_nested_lm_contexts(self) -> None:
        """Test nested context managers with different LMs."""
        global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
        settings.configure(lm=global_lm)

        mock_client1 = AsyncMock(spec=AsyncOpenAI)
        lm1 = OpenAILM(client=mock_client1, default_model="gpt-4o")

        mock_client2 = AsyncMock(spec=AsyncOpenAI)
        lm2 = OpenAILM(client=mock_client2, default_model="gpt-4-turbo")

        with settings.context(lm=lm1):
            assert settings.lm == lm1

            with settings.context(lm=lm2):
                assert settings.lm == lm2

            # Back to outer context
            assert settings.lm == lm1

        # Back to global
        assert settings.lm == global_lm

    def test_context_preserves_lm_when_only_changing_other_settings(self) -> None:
        """Test that LM is preserved when context only changes kwargs."""
        global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
        settings.configure(lm=global_lm)
        original_lm = settings.lm

        # Only changing kwargs should keep the same LM
        with settings.context(temperature=0.9):
            assert settings.lm is original_lm

        # Providing a new LM creates a new LM
        new_lm = LM(model="gpt-4", api_key="sk-test")
        with settings.context(lm=new_lm):
            assert settings.lm is not original_lm

        # After all contexts, should be back to original LM
        assert settings.lm is original_lm


class TestLMAbstraction:
    """Tests for LM abstraction interface."""

    def test_lm_is_abstract(self) -> None:
        """Test that BaseLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLM()  # type: ignore[abstract]

    def test_custom_lm_implementation(self) -> None:
        """Test that custom LM implementations work."""

        class MockLM(BaseLM):
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            async def acomplete(
                self,
                messages: list[dict[str, Any]],
                *,
                model: str,
                tools: list[dict[str, Any]] | None = None,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict[str, Any]:
                # Record the call
                self.calls.append(
                    {
                        "messages": messages,
                        "model": model,
                        "tools": tools,
                        "stream": stream,
                        "kwargs": kwargs,
                    }
                )

                # Return mock response
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Mock response",
                            }
                        }
                    ]
                }

        # Create and use custom LM
        mock_lm = MockLM()
        settings.configure(lm=mock_lm)

        assert settings.lm == mock_lm

        # Could be used in actual predictions (tested elsewhere)
        # Just verify it's accessible
        assert isinstance(settings.lm, BaseLM)
