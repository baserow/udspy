"""Base classes for modules."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from udspy.streaming import Prediction, StreamEvent


class Module:
    """Base class for all udspy modules.

    Modules are composable async-first units. The core method is `astream()`
    which yields StreamEvent objects. Sync wrappers are provided for convenience.

    Subclasses should implement `astream()` to define their behavior.

    Example:
        ```python
        # Async streaming (real-time)
        async for event in module.astream(question="What is AI?"):
            if isinstance(event, StreamChunk):
                print(event.delta, end="", flush=True)
            elif isinstance(event, Prediction):
                result = event

        # Async non-streaming
        result = await module.aforward(question="What is AI?")

        # Sync (for scripts, notebooks)
        result = module(question="What is AI?")
        result = module.forward(question="What is AI?")
        ```
    """

    async def astream(self, **inputs: Any) -> AsyncGenerator[StreamEvent, None]:
        """Core async streaming method. Must be implemented by subclasses.

        This is the fundamental method that all modules must implement.
        It yields StreamEvent objects (including StreamChunk and Prediction).

        Args:
            **inputs: Input values for the module

        Yields:
            StreamEvent objects (StreamChunk for incremental output,
            Prediction for final result, and any custom events)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement astream() method")
        # Make this a generator to match the return type
        yield  # This line will never execute but satisfies the generator type

    async def aforward(self, **inputs: Any) -> Prediction:
        """Async non-streaming method. Consumes astream() and returns final result.

        This is a convenience method that collects all events from astream()
        and returns only the final Prediction. Override this if you need
        custom non-streaming behavior.

        Args:
            **inputs: Input values for the module

        Returns:
            Final Prediction object
        """
        async for event in self.astream(**inputs):
            if isinstance(event, Prediction):
                return event

        raise RuntimeError(f"{self.__class__.__name__}.astream() did not yield a Prediction")

    def forward(self, **inputs: Any) -> Prediction:
        """Sync non-streaming method. Wraps aforward() with async_to_sync.

        This provides sync compatibility for scripts and notebooks. Cannot be
        called from within an async context (use aforward() instead).

        Args:
            **inputs: Input values for the module

        Returns:
            Final Prediction object

        Raises:
            RuntimeError: If called from within an async context
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                f"Cannot call {self.__class__.__name__}.forward() from async context. "
                f"Use 'await {self.__class__.__name__[0].lower() + self.__class__.__name__[1:]}.aforward(...)' instead."
            )
        except RuntimeError as e:
            # No running loop - we're in sync context, proceed
            if "no running event loop" not in str(e).lower():
                raise

        # Run async code from sync context
        return asyncio.run(self.aforward(**inputs))

    def __call__(self, **inputs: Any) -> Prediction:
        """Sync convenience method. Calls forward().

        Args:
            **inputs: Input values for the module

        Returns:
            Final Prediction object
        """
        return self.forward(**inputs)
