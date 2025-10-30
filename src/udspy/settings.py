"""Global settings and configuration."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from udspy.lm import LM, BaseLM


class Settings:
    """Global settings for udspy.

    udspy uses a single LM (Language Model) instance to handle all provider interactions.
    Create an LM using the factory function and configure it globally or per-context.
    """

    def __init__(self) -> None:
        self._lm: BaseLM | None = None
        self._default_kwargs: dict[str, Any] = {}
        self._callbacks: list[Any] = []

        # Context-specific overrides (thread-safe)
        self._context_lm: ContextVar[BaseLM | None] = ContextVar("context_lm", default=None)
        self._context_kwargs: ContextVar[dict[str, Any] | None] = ContextVar(
            "context_kwargs", default=None
        )
        self._context_callbacks: ContextVar[list[Any] | None] = ContextVar(
            "context_callbacks", default=None
        )

    def configure(
        self,
        lm: BaseLM | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure global language model and defaults.

        If lm is not provided but model is, creates LM from environment variables:
        - api_key: UDSPY_LM_API_KEY or OPENAI_API_KEY
        - model: UDSPY_LM_MODEL
        - base_url: UDSPY_LM_BASE_URL

        Args:
            lm: Language model instance. If not provided, creates from model/api_key/base_url
            model: Model identifier (e.g., "gpt-4o", "groq/llama-3"). Used if lm not provided
            api_key: API key. If not provided, reads from UDSPY_LM_API_KEY or OPENAI_API_KEY
            base_url: Base URL. If not provided, reads from UDSPY_LM_BASE_URL
            callbacks: List of callback handlers for telemetry/monitoring
            **kwargs: Default kwargs for all completions (temperature, etc.)

        Examples:
            # From environment variables (recommended)
            # Set: UDSPY_LM_MODEL=gpt-4o, UDSPY_LM_API_KEY=sk-...
            udspy.settings.configure()

            # From explicit parameters
            udspy.settings.configure(model="gpt-4o", api_key="sk-...")

            # With custom LM instance
            from udspy import LM
            lm = LM(model="gpt-4o", api_key="sk-...")
            udspy.settings.configure(lm=lm)

            # With Groq
            lm = LM(model="groq/llama-3-70b", api_key="gsk-...")
            udspy.settings.configure(lm=lm)

            # With Ollama (local)
            lm = LM(model="ollama/llama2")
            udspy.settings.configure(lm=lm)

            # With callbacks
            from udspy import BaseCallback

            class LoggingCallback(BaseCallback):
                def on_lm_start(self, call_id, instance, inputs):
                    print(f"LLM called: {inputs}")

            udspy.settings.configure(
                model="gpt-4o",
                api_key="sk-...",
                callbacks=[LoggingCallback()]
            )
        """
        if lm:
            self._lm = lm
        else:
            # Create LM from parameters or environment variables
            if model is None:
                model = os.getenv("UDSPY_LM_MODEL")
            if api_key is None:
                api_key = os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if base_url is None:
                base_url = os.getenv("UDSPY_LM_BASE_URL")

            if not model:
                raise ValueError(
                    "No model specified. Either provide lm= or model=, "
                    "or set UDSPY_LM_MODEL environment variable."
                )

            # Create LM using factory
            self._lm = LM(model=model, api_key=api_key, base_url=base_url)

        if callbacks is not None:
            self._callbacks = callbacks

        self._default_kwargs.update(kwargs)

    @property
    def lm(self) -> BaseLM:
        """Get the language model instance (context-aware).

        This is the standard way to access the LM for predictions.

        Returns:
            LM instance for making predictions

        Raises:
            RuntimeError: If LM not configured
        """
        # Check context first
        context_lm = self._context_lm.get()
        if context_lm is not None:
            return context_lm

        # Fall back to global LM
        if self._lm is None:
            raise RuntimeError(
                "LM not configured. Call udspy.settings.configure() first.\n"
                "Example: udspy.settings.configure(model='gpt-4o', api_key='sk-...')"
            )
        return self._lm

    @property
    def callbacks(self) -> list[Any]:
        """Get the default callbacks (context-aware)."""
        # Check context first
        context_callbacks = self._context_callbacks.get()
        if context_callbacks is not None:
            return context_callbacks

        return self._callbacks

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Get the default kwargs for completions (context-aware)."""
        # Start with global defaults
        result = self._default_kwargs.copy()

        # Override with context-specific kwargs if present
        context_kwargs = self._context_kwargs.get()
        if context_kwargs is not None:
            result.update(context_kwargs)

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key (for callback compatibility).

        Args:
            key: Setting key to retrieve
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        if key == "callbacks":
            # Check context first
            context_callbacks = self._context_callbacks.get()
            if context_callbacks is not None:
                return context_callbacks
            return self._callbacks
        return default

    @contextmanager
    def context(
        self,
        lm: BaseLM | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Context manager for temporary settings overrides.

        This is thread-safe and allows you to use different LMs or settings
        within a specific context. Useful for multi-tenant applications.

        Args:
            lm: Temporary LM instance. If not provided, creates from model/api_key/base_url
            model: Temporary model identifier
            api_key: Temporary API key
            base_url: Temporary base URL
            callbacks: Temporary callback handlers
            **kwargs: Temporary kwargs for completions

        Examples:
            # Global settings
            udspy.settings.configure(model="gpt-4o-mini", api_key="global-key")

            class QA(Signature):
                question: str = InputField()
                answer: str = OutputField()

            predictor = Predict(QA)

            # Temporary override for specific context
            with udspy.settings.context(model="gpt-4", api_key="tenant-key"):
                result = predictor(question="...")  # Uses gpt-4 with tenant-key

            # Back to global settings
            result = predictor(question="...")  # Uses gpt-4o-mini with global-key

            # With custom LM
            from udspy import LM

            with udspy.settings.context(lm=LM(model="groq/llama-3-70b", api_key="...")):
                result = predictor(question="...")  # Uses Groq
        """
        # Save current context values
        prev_lm = self._context_lm.get()
        prev_kwargs = self._context_kwargs.get()
        prev_callbacks = self._context_callbacks.get()

        try:
            # Set context-specific LM
            if lm:
                self._context_lm.set(lm)
            elif model or api_key or base_url:
                # Create temporary LM
                temp_model = model or os.getenv("UDSPY_LM_MODEL") or "gpt-4o"
                temp_lm = LM(model=temp_model, api_key=api_key, base_url=base_url)
                self._context_lm.set(temp_lm)

            if callbacks is not None:
                self._context_callbacks.set(callbacks)

            if kwargs:
                # Merge with previous context kwargs if any
                merged_kwargs = (prev_kwargs or {}).copy()
                merged_kwargs.update(kwargs)
                self._context_kwargs.set(merged_kwargs)

            yield

        finally:
            # Restore previous context values
            self._context_lm.set(prev_lm)
            self._context_kwargs.set(prev_kwargs)
            self._context_callbacks.set(prev_callbacks)


# Global settings instance
settings = Settings()
