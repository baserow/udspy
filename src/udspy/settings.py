"""Global settings and configuration."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from openai import AsyncOpenAI, OpenAI


class Settings:
    """Global settings for udspy."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._default_model: str = "gpt-4o-mini"
        self._default_kwargs: dict[str, Any] = {}

        # Context-specific overrides (thread-safe)
        self._context_client: ContextVar[OpenAI | None] = ContextVar(
            "context_client", default=None
        )
        self._context_async_client: ContextVar[AsyncOpenAI | None] = ContextVar(
            "context_async_client", default=None
        )
        self._context_model: ContextVar[str | None] = ContextVar("context_model", default=None)
        self._context_kwargs: ContextVar[dict[str, Any] | None] = ContextVar(
            "context_kwargs", default=None
        )

    def configure(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure global OpenAI client and defaults.

        Args:
            api_key: OpenAI API key (creates default client)
            model: Default model to use for all predictions
            client: Custom synchronous OpenAI client
            async_client: Custom asynchronous OpenAI client
            **kwargs: Default kwargs for all chat completions
        """
        if client:
            self._client = client
        elif api_key:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        if async_client:
            self._async_client = async_client
        elif api_key:
            self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        if model:
            self._default_model = model

        self._default_kwargs.update(kwargs)

    @property
    def client(self) -> OpenAI:
        """Get the synchronous OpenAI client (context-aware)."""
        # Check context first
        context_client = self._context_client.get()
        if context_client is not None:
            return context_client

        # Fall back to global client
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not configured. Call udspy.settings.configure() first."
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get the asynchronous OpenAI client (context-aware)."""
        # Check context first
        context_async_client = self._context_async_client.get()
        if context_async_client is not None:
            return context_async_client

        # Fall back to global client
        if self._async_client is None:
            raise RuntimeError(
                "OpenAI client not configured. Call udspy.settings.configure() first."
            )
        return self._async_client

    @property
    def default_model(self) -> str:
        """Get the default model name (context-aware)."""
        # Check context first
        context_model = self._context_model.get()
        if context_model is not None:
            return context_model

        # Fall back to global model
        return self._default_model

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Get the default kwargs for chat completions (context-aware)."""
        # Start with global defaults
        result = self._default_kwargs.copy()

        # Override with context-specific kwargs if present
        context_kwargs = self._context_kwargs.get()
        if context_kwargs is not None:
            result.update(context_kwargs)

        return result

    @contextmanager
    def context(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: OpenAI | None = None,
        async_client: AsyncOpenAI | None = None,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Context manager for temporary settings overrides.

        This is thread-safe and allows you to use different API keys, models,
        or other settings within a specific context.

        Args:
            api_key: Temporary OpenAI API key (creates temporary client)
            model: Temporary model to use
            client: Temporary synchronous OpenAI client
            async_client: Temporary asynchronous OpenAI client
            **kwargs: Temporary kwargs for chat completions

        Example:
            ```python
            # Global settings
            udspy.settings.configure(api_key="global-key", model="gpt-4o-mini")

            # Temporary override for a specific context
            with udspy.settings.context(api_key="other-key", model="gpt-4"):
                predictor = Predict(QA)
                result = predictor(question="...")  # Uses "other-key" and "gpt-4"

            # Back to global settings
            result = predictor(question="...")  # Uses "global-key" and "gpt-4o-mini"
            ```
        """
        # Save current context values
        prev_client = self._context_client.get()
        prev_async_client = self._context_async_client.get()
        prev_model = self._context_model.get()
        prev_kwargs = self._context_kwargs.get()

        try:
            # Set context-specific values
            if client:
                self._context_client.set(client)
            elif api_key:
                self._context_client.set(OpenAI(api_key=api_key))

            if async_client:
                self._context_async_client.set(async_client)
            elif api_key:
                self._context_async_client.set(AsyncOpenAI(api_key=api_key))

            if model:
                self._context_model.set(model)

            if kwargs:
                # Merge with previous context kwargs if any
                merged_kwargs = (prev_kwargs or {}).copy()
                merged_kwargs.update(kwargs)
                self._context_kwargs.set(merged_kwargs)

            yield

        finally:
            # Restore previous context values
            self._context_client.set(prev_client)
            self._context_async_client.set(prev_async_client)
            self._context_model.set(prev_model)
            self._context_kwargs.set(prev_kwargs)


# Global settings instance
settings = Settings()
