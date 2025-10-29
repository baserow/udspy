"""Decorators for module behavior."""

import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from udspy.confirmation import ConfirmationRequired

T = TypeVar("T")


def suspendable(aexecute_fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator that adds suspend/resume support to aexecute() methods.

    This decorator:
    1. Checks for resume_state parameter
    2. If present, calls module.aresume() instead of executing normally
    3. If ConfirmationRequired is raised, calls module.asuspend() to save state
    4. Attaches saved state to the exception before re-raising

    This enables all modules to support suspend/resume uniformly without
    duplicating the logic in each implementation.

    Usage:
        ```python
        class MyModule(Module):
            @suspendable
            async def aexecute(self, *, stream=False, **inputs):
                # Your implementation
                ...

            async def asuspend(self, exception):
                # Save module-specific state
                return {"my_state": ...}

            async def aresume(self, user_response, saved_state):
                # Resume from saved state
                ...
        ```

    Args:
        aexecute_fn: The aexecute method to wrap

    Returns:
        Wrapped aexecute method with suspend/resume support
    """

    @functools.wraps(aexecute_fn)
    async def wrapper(self: Any, *, resume_state: Any = None, **kwargs: Any) -> Any:
        # If resuming from a previous suspension
        if resume_state is not None:
            from udspy.confirmation import ResumeState

            # Support both ResumeState and raw ConfirmationRequired
            if isinstance(resume_state, ResumeState):
                return await self.aresume(resume_state.user_response, resume_state.exception)
            elif isinstance(resume_state, ConfirmationRequired):
                # Backward compatibility: treat raw exception as "yes" response
                return await self.aresume("yes", resume_state)
            else:
                # Assume it's saved state from old API
                return await self.aresume("yes", resume_state)

        # Normal execution - call the actual aexecute
        try:
            return await aexecute_fn(self, **kwargs)
        except ConfirmationRequired as e:
            # Suspend: save state before re-raising
            saved_state = await self.asuspend(e)
            # Attach saved state to exception for convenience
            e.saved_state = saved_state  # type: ignore[attr-defined]
            raise

    return wrapper
