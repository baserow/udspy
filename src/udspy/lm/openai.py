"""OpenAI language model implementation."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional, Union

from udspy.callback import with_callbacks
from udspy.lm.base import LM

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


class OpenAILM(LM):
    """OpenAI language model implementation.

    Wraps AsyncOpenAI client to provide the LM interface.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        default_model: str | None = None,
        client: Optional["AsyncOpenAI"] = None,
        provider: str = "openai",
    ):
        """Initialize OpenAI LM.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API
            default_model: Default model to use if not specified in acomplete()
            client: Optional AsyncOpenAI client (for testing)
            provider: Provider name (e.g., "openai", "bedrock", "ollama")
        """

        from openai import AsyncOpenAI

        self.client = (
            client if client is not None else AsyncOpenAI(api_key=api_key, base_url=base_url)
        )
        self.default_model = default_model
        self.provider = provider

    @property
    def model(self) -> str | None:
        """Get the default model."""
        return self.default_model

    @with_callbacks
    async def _acomplete(
        self, **kwargs: Any
    ) -> Union["ChatCompletion", AsyncGenerator["ChatCompletionChunk", None]]:
        return await self.client.chat.completions.create(**kwargs)

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union["ChatCompletion", AsyncGenerator["ChatCompletionChunk", None]]:
        """Generate completion using OpenAI API.

        Args:
            messages: List of messages in OpenAI format
            model: Model to use (overrides default_model)
            tools: Optional list of tool schemas
            stream: If True, return streaming response
            **kwargs: Additional OpenAI parameters (temperature, max_tokens, etc.)

        Returns:
            ChatCompletion if stream=False, AsyncGenerator[ChatCompletionChunk, None] if stream=True
        """
        from openai import APIError
        from tenacity import (
            AsyncRetrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Use provided model or fall back to default
        actual_model = model or self.default_model
        if not actual_model:
            raise ValueError("No model specified and no default_model set")

        # Build completion kwargs
        completion_kwargs: dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "stream": stream,
            "max_tokens": kwargs.pop("max_tokens", 8000),
            **kwargs,
        }

        # Add tools if provided
        if tools:
            completion_kwargs["tools"] = tools

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(APIError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=0.2, max=3),
        ):
            with attempt:
                return await self._acomplete(**completion_kwargs)

        raise AssertionError("unreachable")  # AsyncRetrying always executes at least once


__all__ = ["OpenAILM"]
