"""Global settings and configuration."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from udspy.lm import BaseLM


class Settings:
    """Global settings for udspy.

    udspy uses a single LM (Language Model) instance to handle all provider interactions.
    Create an LM using the factory function and configure it globally or per-context.
    """

    def __init__(self) -> None:
        self._lm: BaseLM | None = None
        self._default_kwargs: dict[str, Any] = {}
        self._callbacks: list[Any] = []

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
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure global language model and defaults.

        Args:
            lm: Language model instance. If not provided, creates from environment variables
            callbacks: List of callback handlers for telemetry/monitoring
            **kwargs: Default kwargs for all completions (temperature, etc.)

        Examples:
            # From environment variables
            # Set: UDSPY_LM_MODEL=gpt-4o, UDSPY_LM_API_KEY=sk-...
            udspy.settings.configure()

            # With custom LM instance
            from udspy import LM
            lm = LM(model="gpt-4o", api_key="sk-...")
            udspy.settings.configure(lm=lm)

            # With Ollama (local)
            lm = LM(model="ollama/llama2")
            udspy.settings.configure(lm=lm)

            # With callbacks
            from udspy import LM, BaseCallback

            class LoggingCallback(BaseCallback):
                def on_lm_start(self, call_id, instance, inputs):
                    print(f"LLM called: {inputs}")

            lm = LM(model="gpt-4o", api_key="sk-...")
            udspy.settings.configure(lm=lm, callbacks=[LoggingCallback()])
        """
        if lm:
            self._lm = lm

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
        context_lm = self._context_lm.get()
        if context_lm is not None:
            return context_lm

        if self._lm is None:
            raise RuntimeError(
                "LM not configured. Call udspy.settings.configure() first.\n"
                "Example: udspy.settings.configure(lm=LM(model='gpt-4o', api_key='sk-...'))"
            )
        return self._lm

    @property
    def callbacks(self) -> list[Any]:
        """Get the default callbacks (context-aware)."""
        context_callbacks = self._context_callbacks.get()
        if context_callbacks is not None:
            return context_callbacks

        return self._callbacks

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Get the default kwargs for completions (context-aware)."""
        result = self._default_kwargs.copy()

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
            context_callbacks = self._context_callbacks.get()
            if context_callbacks is not None:
                return context_callbacks
            return self._callbacks
        return default

    @contextmanager
    def context(
        self,
        lm: BaseLM | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Context manager for temporary settings overrides.

        This is thread-safe and allows you to use different LMs or settings
        within a specific context. Useful for multi-tenant applications.

        Args:
            lm: Temporary LM instance
            callbacks: Temporary callback handlers
            **kwargs: Temporary kwargs for completions

        Examples:
            # Global settings
            from udspy import LM
            lm = LM(model="gpt-4o-mini", api_key="global-key")
            udspy.settings.configure(lm=lm)

            class QA(Signature):
                question: str = InputField()
                answer: str = OutputField()

            predictor = Predict(QA)

            # Temporary override for specific context
            tenant_lm = LM(model="gpt-4", api_key="tenant-key")
            with udspy.settings.context(lm=tenant_lm):
                result = predictor(question="...")  # Uses gpt-4 with tenant-key

            # Back to global settings
            result = predictor(question="...")  # Uses gpt-4o-mini with global-key

            # With Ollama
            ollama_lm = LM(model="ollama/llama2")
            with udspy.settings.context(lm=ollama_lm):
                result = predictor(question="...")  # Uses Ollama
        """
        prev_lm = self._context_lm.get()
        prev_kwargs = self._context_kwargs.get()
        prev_callbacks = self._context_callbacks.get()

        try:
            if lm:
                self._context_lm.set(lm)

            if callbacks is not None:
                self._context_callbacks.set(callbacks)

            if kwargs:
                merged_kwargs = (prev_kwargs or {}).copy()
                merged_kwargs.update(kwargs)
                self._context_kwargs.set(merged_kwargs)

            yield

        finally:
            self._context_lm.set(prev_lm)
            self._context_kwargs.set(prev_kwargs)
            self._context_callbacks.set(prev_callbacks)


settings = Settings()
