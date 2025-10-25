"""Utility functions for udspy."""

import asyncio
import functools
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def asyncify(func: Callable[..., T]) -> Callable[..., T]:
    """Convert a sync function to async using thread pool execution.

    This is useful for calling sync I/O operations (database queries, file
    operations, network requests) from async context without blocking the
    event loop.

    If the function is already async, it's returned as-is.

    Args:
        func: A sync or async function to wrap

    Returns:
        An async function that executes the original in a thread pool
        (if sync) or returns the original function (if already async)

    Example:
        ```python
        # With Django ORM
        from django.contrib.auth.models import User
        from udspy.utils import asyncify

        @asyncify
        def get_user(user_id: int):
            return User.objects.get(id=user_id)

        # Use in async context
        async def my_view():
            user = await get_user(123)
            return user.username

        # With SQLAlchemy
        from sqlalchemy.orm import Session
        from udspy.utils import asyncify

        @asyncify
        def query_users(session: Session, name: str):
            return session.query(User).filter(User.name == name).all()

        async def search():
            users = await query_users(session, "Alice")
            return users

        # With file I/O
        @asyncify
        def read_large_file(path: str):
            with open(path) as f:
                return f.read()

        async def process_file():
            content = await read_large_file("/path/to/file.txt")
            return content
        ```

    Note:
        - This uses Python's thread pool, so it doesn't help with CPU-bound work
        - For CPU-bound tasks, consider using ProcessPoolExecutor instead
        - This is not Django-specific; it works with any sync code
    """
    # If already async, return as-is
    if asyncio.iscoroutinefunction(func):
        return func  # type: ignore[return-value]

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        """Async wrapper that runs sync function in thread pool."""
        return await asyncio.to_thread(func, *args, **kwargs)

    return async_wrapper  # type: ignore[return-value]


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


__all__ = ["asyncify", "ensure_sync_context"]
