"""Async utility functions."""

import asyncio
import contextvars
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any


def ensure_sync_context(method_name: str) -> None:
    """Ensure we're NOT in an async context (for sync methods).

    This prevents calling sync methods like forward() or resume() from within
    async code, which would fail because asyncio.run() can't be called when
    there's already a running event loop.

    Args:
        method_name: Name of the method being called (for error message)

    Raises:
        RuntimeError: If called from within an async context, with a helpful
            message suggesting the async alternative

    Example:
        ```python
        def forward(self, **inputs):
            ensure_sync_context("forward")  # Check first
            return asyncio.run(self.aforward(**inputs))
        ```

    Note:
        This uses asyncio.get_running_loop() which raises RuntimeError
        if there's no event loop. We catch that specific error to allow
        the sync method to proceed.
    """
    try:
        asyncio.get_running_loop()
        # If we get here, there IS a running loop - we're in async context
        # Extract class name from method_name if it's in "ClassName.method" format
        if "." in method_name:
            class_name, method = method_name.rsplit(".", 1)
            async_method = f"await {class_name[0].lower() + class_name[1:]}.a{method}(...)"
        else:
            async_method = f"await ...a{method_name}(...)"

        raise RuntimeError(
            f"Cannot call {method_name}() from async context. Use '{async_method}' instead."
        )
    except RuntimeError as e:
        # Check if it's the "no running event loop" error (which is what we want)
        if "no running event loop" not in str(e).lower():
            # It's a different RuntimeError - re-raise it
            raise


async def execute_function_async(
    func: Callable[..., Any], kwargs: dict[str, Any] | None = None
) -> Any:
    """Execute a function asynchronously, handling both sync and async functions.

    Args:
        func: Function to execute (can be sync or async)
        kwargs: Keyword arguments to pass

    Returns:
        Function result
    """

    kwargs = kwargs or {}
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    else:
        ctx = contextvars.copy_context()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(ctx.run, lambda: func(**kwargs)))


def run_async_with_context(coro: Any) -> Any:
    """Run an async coroutine with the current context preserved.

    This is useful when you need to use asyncio.run() but want to preserve
    contextvars (like settings._context_lm) from the calling context.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Example:
        ```python
        import contextvars

        # Define a context variable
        my_var = contextvars.ContextVar("my_var", default=None)
        my_var.set("important_value")

        # This preserves the context:
        result = run_async_with_context(some_async_func())
        # Inside some_async_func(), my_var.get() returns "important_value"
        ```

    Note:
        This is the opposite of execute_function_async():
        - execute_function_async: calls sync from async, preserving context
        - run_async_with_context: calls async from sync, preserving context

        Both use the same pattern: copy_context() + ctx.run()
    """
    ctx = contextvars.copy_context()
    return ctx.run(asyncio.run, coro)


__all__ = [
    "ensure_sync_context",
    "execute_function_async",
    "run_async_with_context",
]
