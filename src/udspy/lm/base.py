"""Base language model abstraction."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class LM(ABC):
    """Abstract base class for language model providers.

    This abstraction allows udspy to work with different LLM providers
    (OpenAI, Anthropic, local models, etc.) through a common interface.

    Implementations should handle:
    - API calls to the provider
    - Response format normalization
    - Streaming support
    - Error handling and retries

    Usage:
        ```python
        # Async usage
        response = await lm.acomplete(messages, model="gpt-4o")

        # Sync usage
        response = lm.complete(messages, model="gpt-4o")

        # Callable (sync)
        response = lm(messages, model="gpt-4o")
        ```
    """

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        """Generate a completion from the language model.

        Args:
            messages: List of messages in OpenAI format
                [{"role": "system", "content": "..."}, ...]
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
            tools: Optional list of tool schemas in OpenAI format
            stream: If True, return an async generator of chunks
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            If stream=False: Completion response object
            If stream=True: AsyncGenerator yielding completion chunks

        Raises:
            LMError: On API errors, rate limits, etc.
        """
        pass

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Synchronous version of acomplete.

        Args:
            messages: List of messages in OpenAI format
            model: Model identifier
            tools: Optional list of tool schemas
            stream: If True, return an async generator (must be consumed with async for)
            **kwargs: Provider-specific parameters

        Returns:
            Completion response object
        """
        return asyncio.run(
            self.acomplete(messages, model=model, tools=tools, stream=stream, **kwargs)
        )

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Make LM callable - delegates to complete().

        Args:
            messages: List of messages in OpenAI format
            model: Model identifier
            tools: Optional list of tool schemas
            stream: If True, return an async generator
            **kwargs: Provider-specific parameters

        Returns:
            Completion response object

        Example:
            ```python
            from udspy import LM

            lm = LM(model="gpt-4o", api_key="...")
            response = lm(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o"
            )
            ```
        """
        return self.complete(messages, model=model, tools=tools, stream=stream, **kwargs)


__all__ = ["LM"]
