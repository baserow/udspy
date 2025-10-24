"""Global settings and configuration."""

from typing import Any

from openai import AsyncOpenAI, OpenAI


class Settings:
    """Global settings for udspy."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._default_model: str = "gpt-4o-mini"
        self._default_kwargs: dict[str, Any] = {}

    def configure(
        self,
        api_key: str | None = None,
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
            self._client = OpenAI(api_key=api_key)

        if async_client:
            self._async_client = async_client
        elif api_key:
            self._async_client = AsyncOpenAI(api_key=api_key)

        if model:
            self._default_model = model

        self._default_kwargs.update(kwargs)

    @property
    def client(self) -> OpenAI:
        """Get the synchronous OpenAI client."""
        if self._client is None:
            raise RuntimeError(
                "OpenAI client not configured. Call udspy.settings.configure() first."
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get the asynchronous OpenAI client."""
        if self._async_client is None:
            raise RuntimeError(
                "OpenAI client not configured. Call udspy.settings.configure() first."
            )
        return self._async_client

    @property
    def default_model(self) -> str:
        """Get the default model name."""
        return self._default_model

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Get the default kwargs for chat completions."""
        return self._default_kwargs.copy()


# Global settings instance
settings = Settings()
