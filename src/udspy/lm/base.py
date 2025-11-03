"""Base language model abstraction."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, overload

from udspy.callback import with_callbacks


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
        # Simple usage (uses default model)
        answer = lm("How are you?")

        # Async usage
        response = await lm.acomplete(messages, model="gpt-4o")

        # Sync usage with full control
        response = lm.complete(messages, model="gpt-4o")
        ```
    """

    @property
    def model(self) -> str | None:
        """Get the default model for this LM instance.

        Implementations should override this to provide their default model.
        """
        return None

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        """Generate a completion from the language model.

        Args:
            messages: List of messages in OpenAI format
                [{"role": "system", "content": "..."}, ...]
            model: Model identifier (e.g., "gpt-4o"). If None, uses default model.
            tools: Optional list of tool schemas in OpenAI format
            stream: If True, return an async generator of chunks
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            If stream=False: Completion response object
            If stream=True: AsyncGenerator yielding completion chunks

        Raises:
            LMError: On API errors, rate limits, etc.

        Note:
            Implementations should add @with_callbacks decorator to enable
            automatic callback invocation.
        """
        pass

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Synchronous version of acomplete.

        Args:
            messages: List of messages in OpenAI format
            model: Model identifier (defaults to self.model)
            tools: Optional list of tool schemas
            stream: If True, return an async generator (must be consumed with async for)
            **kwargs: Provider-specific parameters

        Returns:
            Completion response object
        """
        return asyncio.run(
            self.acomplete(messages, model=model, tools=tools, stream=stream, **kwargs)
        )

    @overload
    async def acall(
        self,
        prompt: str,
        *,
        model: str | None = None,
        tools: None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str: ...

    @overload
    async def acall(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    @with_callbacks
    async def acall(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Any:
        """Async version of __call__ with callback support.

        Supports two modes:
        1. Simple string prompt: Returns just the text content
        2. Full messages list: Returns complete response object

        Args:
            prompt_or_messages: Either a simple string prompt or list of messages
            model: Model identifier (defaults to self.model)
            tools: Optional list of tool schemas (ignored for string prompts)
            stream: If True, return an async generator
            **kwargs: Provider-specific parameters

        Returns:
            If prompt is a string: Just the text content (str)
            If messages is a list: Complete response object (Any)

        Examples:
            ```python
            # Simple usage - returns text directly
            answer = await lm.acall("How are you?")
            print(answer)  # "I'm doing well, thanks!"

            # With model override
            answer = await lm.acall("Explain quantum physics", model="gpt-4")

            # Full control - returns response object
            response = await lm.acall(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                tools=[...],
            )
            ```
        """
        if isinstance(prompt_or_messages, str):
            # Ignore tools parameter for string prompts (already enforced by overload)
            messages = [{"role": "user", "content": prompt_or_messages}]
            response = await self.acomplete(messages, model=model, stream=stream, **kwargs)
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "content") and message.content:
                    return message.content
            return str(response)
        else:
            return await self.acomplete(
                prompt_or_messages,
                model=model,
                tools=tools,
                stream=stream,
                **kwargs,
            )

    @overload
    def __call__(
        self,
        prompt: str,
        *,
        model: str | None = None,
        tools: None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    @with_callbacks
    def __call__(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Any:
        """Make LM callable for convenient usage with callback support.

        Supports two modes:
        1. Simple string prompt: Returns just the text content
        2. Full messages list: Returns complete response object

        Args:
            prompt_or_messages: Either a simple string prompt or list of messages
            model: Model identifier (defaults to self.model)
            tools: Optional list of tool schemas (ignored for string prompts)
            stream: If True, return an async generator
            **kwargs: Provider-specific parameters

        Returns:
            If prompt is a string: Just the text content (str)
            If messages is a list: Complete response object (Any)

        Examples:
            ```python
            # Simple usage - returns text directly
            answer = lm("How are you?")
            print(answer)  # "I'm doing well, thanks!"

            # With model override
            answer = lm("Explain quantum physics", model="gpt-4")

            # Full control - returns response object
            response = lm(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
                tools=[...],
            )
            ```
        """
        if isinstance(prompt_or_messages, str):
            # Ignore tools parameter for string prompts (already enforced by overload)
            messages = [{"role": "user", "content": prompt_or_messages}]
            response = self.complete(messages, model=model, stream=stream, **kwargs)
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "content") and message.content:
                    return message.content
            return str(response)
        else:
            return self.complete(
                prompt_or_messages,
                model=model,
                tools=tools,
                stream=stream,
                **kwargs,
            )


__all__ = ["LM"]
