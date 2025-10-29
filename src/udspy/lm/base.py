"""Base language model abstraction."""

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


__all__ = ["LM"]
