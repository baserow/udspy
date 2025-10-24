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


__all__ = ["asyncify"]
