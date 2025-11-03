"""Groq language model implementation using the official groq library."""

from typing import Any

from groq import AsyncGroq, AsyncStream
from groq.types.chat import ChatCompletion, ChatCompletionChunk

from udspy.lm.base import LM


class GroqLM(LM):
    """Groq language model implementation.

    Uses the official groq library to interact with Groq's API.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        default_model: str | None = None,
        client: AsyncGroq | None = None,
    ):
        """Initialize Groq LM.

        Args:
            api_key: Groq API key (starts with 'gsk_')
            base_url: Optional custom base URL for Groq API
            default_model: Default model to use if not specified in acomplete()
            client: Optional AsyncGroq client (for testing)
        """
        self.client = (
            client if client is not None else AsyncGroq(api_key=api_key, base_url=base_url)
        )
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
        """Generate completion using Groq API.

        Args:
            messages: List of messages in OpenAI format
            model: Model to use (overrides default_model)
                Popular models: llama-3.1-70b-versatile, llama-3.1-8b-instant,
                mixtral-8x7b-32768, gemma2-9b-it
            tools: Optional list of tool schemas
            stream: If True, return streaming response
            **kwargs: Additional Groq parameters (temperature, max_tokens, etc.)

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

        # Call Groq API
        response = await self.client.chat.completions.create(**completion_kwargs)

        return response


__all__ = ["GroqLM"]
