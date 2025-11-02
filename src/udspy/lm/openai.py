"""OpenAI language model implementation."""

from typing import Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from udspy.lm.base import LM


class OpenAILM(LM):
    """OpenAI language model implementation.

    Wraps AsyncOpenAI client to provide the LM interface.
    """

    def __init__(self, client: AsyncOpenAI, default_model: str | None = None):
        """Initialize OpenAI LM.

        Args:
            client: Configured AsyncOpenAI client
            default_model: Default model to use if not specified in acomplete()
        """
        self.client = client
        self.default_model = default_model

    @property
    def model(self) -> str | None:
        """Get the default model."""
        return self.default_model

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Generate completion using OpenAI API.

        Args:
            messages: List of messages in OpenAI format
            model: Model to use (overrides default_model)
            tools: Optional list of tool schemas
            stream: If True, return streaming response
            **kwargs: Additional OpenAI parameters (temperature, max_tokens, etc.)

        Returns:
            ChatCompletion or AsyncStream[ChatCompletionChunk]
        """
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

        # Call OpenAI API
        response = await self.client.chat.completions.create(**completion_kwargs)

        return response


__all__ = ["OpenAILM"]
