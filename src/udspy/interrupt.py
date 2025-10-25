"""Interrupt system for human-in-the-loop interactions."""

import functools
import inspect
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

# Thread-safe and asyncio task-safe interrupt context
_interrupt_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "interrupt_context", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


class ToolCall:
    """Information about a tool call that triggered an interrupt.

    This is optional - interrupts can occur without tool calls.
    """

    def __init__(
        self,
        name: str,
        args: dict[str, Any],
        call_id: str | None = None,
    ):
        """Initialize tool call information.

        Args:
            name: Tool name
            args: Tool arguments
            call_id: Optional tool call ID for tracking
        """
        self.name = name
        self.args = args
        self.call_id = call_id


class HumanInTheLoopRequired(Exception):
    """Raised when human input is needed to proceed.

    This exception pauses execution and allows modules to save state for resumption.
    It can be raised by:
    - Tools decorated with @interruptible
    - Modules that need user input (e.g., ask_to_user)
    - Custom code requiring human interaction

    Attributes:
        question: The question being asked to the user
        interrupt_id: Unique ID for this interrupt
        tool_call: Optional ToolCall information if raised by a tool
        context: General-purpose context dictionary for module state
    """

    def __init__(
        self,
        question: str,
        *,
        interrupt_id: str | None = None,
        tool_call: ToolCall | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize HumanInTheLoopRequired exception.

        Args:
            question: Question to ask the user
            interrupt_id: Unique ID for this interrupt (auto-generated if not provided)
            tool_call: Optional tool call information
            context: Optional context dictionary for module-specific state
        """
        super().__init__(question)
        self.question = question
        self.interrupt_id = interrupt_id or str(uuid.uuid4())
        self.tool_call = tool_call
        self.context = context or {}


def get_interrupt_context() -> dict[str, Any]:
    """Get the current interrupt context.

    Returns:
        Dictionary mapping interrupt IDs to their approval status/data
    """
    ctx = _interrupt_context.get()
    return ctx.copy() if ctx is not None else {}


def set_interrupt_approval(
    interrupt_id: str,
    approved: bool = True,
    data: Any = None,
    status: str | None = None,
) -> None:
    """Mark an interrupt as approved in the context.

    Args:
        interrupt_id: The interrupt ID to approve
        approved: Whether the interrupt is approved
        data: Optional data associated with the approval (e.g., modified args)
        status: Optional status string (e.g., "approved", "rejected", "edited", "feedback")
    """
    ctx = _interrupt_context.get()
    if ctx is None:
        ctx = {}
    else:
        ctx = ctx.copy()
    ctx[interrupt_id] = {
        "approved": approved,
        "data": data,
        "status": status or ("approved" if approved else "rejected"),
    }
    _interrupt_context.set(ctx)


def get_interrupt_status(interrupt_id: str) -> str | None:
    """Get the status of an interrupt.

    Args:
        interrupt_id: The interrupt ID to check

    Returns:
        Status string or None if interrupt not found.
        Possible values: "pending", "approved", "rejected", "edited", "feedback"
    """
    ctx = _interrupt_context.get()
    if ctx is None or interrupt_id not in ctx:
        return "pending"
    return ctx[interrupt_id].get(
        "status", "approved" if ctx[interrupt_id].get("approved") else "rejected"
    )


def clear_interrupt(interrupt_id: str) -> None:
    """Remove an interrupt from the context.

    Args:
        interrupt_id: The interrupt ID to clear
    """
    ctx = _interrupt_context.get()
    if ctx is None:
        return
    ctx = ctx.copy()
    ctx.pop(interrupt_id, None)
    _interrupt_context.set(ctx)


def clear_all_interrupts() -> None:
    """Clear all interrupts from the context."""
    _interrupt_context.set({})


def interruptible(func: F) -> F:
    """Decorator that makes a function interruptible.

    When called, the decorator:
    1. Checks if an interrupt ID exists in the context
    2. If approved: proceeds with execution
    3. If not approved: raises HumanInTheLoopRequired

    The interrupt ID is generated based on function name and arguments to ensure
    the same call can be resumed.

    Example:
        ```python
        @interruptible
        def delete_file(path: str) -> str:
            os.remove(path)
            return f"Deleted {path}"

        # First call - raises HumanInTheLoopRequired
        try:
            delete_file("/tmp/test.txt")
        except HumanInTheLoopRequired as e:
            # User confirms
            set_interrupt_approval(e.interrupt_id, approved=True)
            # Retry - now proceeds
            result = delete_file("/tmp/test.txt")
        ```

    Args:
        func: The function to make interruptible

    Returns:
        Wrapped function that checks interrupt context
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate interrupt ID based on function and args
        import json

        func_name = func.__name__

        # Convert positional args to kwargs for consistent handling
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        all_kwargs = dict(bound_args.arguments)

        # Create stable ID from function name and arguments
        args_repr = json.dumps({"kwargs": all_kwargs}, sort_keys=True, default=str)
        interrupt_id = f"{func_name}:{hash(args_repr)}"

        # Check if this interrupt is approved
        ctx = _interrupt_context.get()
        approval = ctx.get(interrupt_id) if ctx is not None else None

        if approval and approval.get("approved"):
            # Approved - proceed with execution
            # Use modified data if provided, otherwise use original kwargs
            if approval.get("data"):
                all_kwargs = approval["data"]

            # Execute the function with kwargs only (no positional args)
            if inspect.iscoroutinefunction(func):
                result = await func(**all_kwargs)
            else:
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**all_kwargs))

            # Clear the interrupt after successful execution
            clear_interrupt(interrupt_id)
            return result
        else:
            # Not approved - raise interrupt exception
            raise HumanInTheLoopRequired(
                question=f"Confirm execution of {func_name} with args: {json.dumps(all_kwargs)}? (yes/no/feedback/edit)",
                interrupt_id=interrupt_id,
                tool_call=ToolCall(name=func_name, args=all_kwargs),
            )

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # For sync functions, we still need to check interrupt context
        import json

        func_name = func.__name__

        # Convert positional args to kwargs for consistent handling
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        all_kwargs = dict(bound_args.arguments)

        args_repr = json.dumps({"kwargs": all_kwargs}, sort_keys=True, default=str)
        interrupt_id = f"{func_name}:{hash(args_repr)}"

        ctx = _interrupt_context.get()
        approval = ctx.get(interrupt_id) if ctx is not None else None

        if approval and approval.get("approved"):
            # Approved - proceed with execution
            if approval.get("data"):
                all_kwargs = approval["data"]

            result = func(**all_kwargs)
            clear_interrupt(interrupt_id)
            return result
        else:
            # Not approved - raise interrupt exception
            raise HumanInTheLoopRequired(
                question=f"Confirm execution of {func_name} with args: {json.dumps(all_kwargs)}? (yes/no/feedback/edit)",
                interrupt_id=interrupt_id,
                tool_call=ToolCall(name=func_name, args=all_kwargs),
            )

    # Return appropriate wrapper based on whether function is async
    if inspect.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore
