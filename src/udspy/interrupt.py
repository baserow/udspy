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


class InterruptRejected(Exception):
    """Raised when user rejects an interrupt/tool execution.

    This signals that the user explicitly does not want the operation to proceed,
    as opposed to just pending approval. This allows calling code to distinguish
    between "waiting for approval" and "user said no".

    Example:
        ```python
        from udspy import interruptible, HumanInTheLoopRequired, InterruptRejected
        from udspy import set_interrupt_approval

        @interruptible
        def delete_database() -> str:
            return "Database deleted"

        try:
            delete_database()
        except HumanInTheLoopRequired as e:
            # User rejects the dangerous operation
            set_interrupt_approval(e.interrupt_id, approved=False, status="rejected")

        try:
            delete_database()  # Try again
        except InterruptRejected as e:
            print(f"Operation stopped: {e.message}")
            # Can handle rejection differently than pending approval
        ```

    Attributes:
        message: Description of what was rejected
        interrupt_id: The interrupt ID that was rejected
        tool_call: Optional ToolCall information if raised by a tool
    """

    def __init__(
        self,
        message: str,
        *,
        interrupt_id: str | None = None,
        tool_call: ToolCall | None = None,
    ):
        """Initialize InterruptRejected exception.

        Args:
            message: Description of what was rejected
            interrupt_id: The interrupt ID that was rejected
            tool_call: Optional tool call information
        """
        super().__init__(message)
        self.message = message
        self.interrupt_id = interrupt_id
        self.tool_call = tool_call


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


def _generate_interrupt_id(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Generate a stable interrupt ID from function name and arguments.

    Creates a unique identifier for an interrupt based on the function being called
    and its arguments. This allows the same function call to be resumed after interruption.

    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (interrupt_id, normalized_kwargs) where:
        - interrupt_id: Stable ID in format "function_name:hash(args)"
        - normalized_kwargs: All arguments converted to kwargs dict
    """
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

    return interrupt_id, all_kwargs


def _handle_interrupt_approval(
    interrupt_id: str, func_name: str, all_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Check interrupt approval status and handle accordingly.

    This function checks if an interrupt has been approved, rejected, or is pending.
    Based on the status, it either:
    - Returns modified kwargs if approved (possibly with user edits)
    - Raises InterruptRejected if user explicitly rejected
    - Raises HumanInTheLoopRequired if pending approval

    Args:
        interrupt_id: The interrupt ID to check
        func_name: Name of the function being called
        all_kwargs: Normalized keyword arguments for the function

    Returns:
        Modified kwargs to use for execution (if approved)

    Raises:
        InterruptRejected: If user rejected the interrupt
        HumanInTheLoopRequired: If interrupt is pending approval
    """
    import json

    ctx = _interrupt_context.get()
    approval = ctx.get(interrupt_id) if ctx is not None else None

    if approval:
        is_approved = approval.get("approved", False)
        status = approval.get("status", "pending")

        if is_approved:
            # Use modified args if provided, otherwise original
            return approval.get("data") or all_kwargs
        elif status == "rejected":
            # User explicitly rejected
            raise InterruptRejected(
                message=f"User rejected execution of {func_name}",
                interrupt_id=interrupt_id,
                tool_call=ToolCall(name=func_name, args=all_kwargs),
            )

    # No approval yet - ask user
    raise HumanInTheLoopRequired(
        question=f"Confirm execution of {func_name} with args: {json.dumps(all_kwargs)}? (yes/no/feedback/edit)",
        interrupt_id=interrupt_id,
        tool_call=ToolCall(name=func_name, args=all_kwargs),
    )


async def _execute_function_async(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Execute a function asynchronously, handling both sync and async functions.

    Args:
        func: Function to execute (can be sync or async)
        kwargs: Keyword arguments to pass

    Returns:
        Function result
    """
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    else:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))


def interruptible(func: F) -> F:
    """Decorator that makes a function interruptible.

    When called, the decorator checks the interrupt context and behaves as follows:
    1. **Pending** (no approval): Raises HumanInTheLoopRequired
    2. **Approved**: Proceeds with execution (possibly with modified args)
    3. **Rejected**: Raises InterruptRejected

    The interrupt ID is generated based on function name and arguments to ensure
    the same call can be resumed after approval.

    Example (Approval Flow):
        ```python
        from udspy import interruptible, HumanInTheLoopRequired, set_interrupt_approval

        @interruptible
        def delete_file(path: str) -> str:
            os.remove(path)
            return f"Deleted {path}"

        # First call - pending, raises HumanInTheLoopRequired
        try:
            delete_file("/tmp/test.txt")
        except HumanInTheLoopRequired as e:
            print(f"Confirm: {e.question}")
            # User approves
            set_interrupt_approval(e.interrupt_id, approved=True)

        # Retry - approved, executes normally
        result = delete_file("/tmp/test.txt")
        print(result)  # "Deleted /tmp/test.txt"
        ```

    Example (Rejection Flow):
        ```python
        from udspy import InterruptRejected

        try:
            delete_file("/tmp/important.txt")
        except HumanInTheLoopRequired as e:
            # User rejects
            set_interrupt_approval(e.interrupt_id, approved=False, status="rejected")

        # Retry - rejected, raises InterruptRejected
        try:
            delete_file("/tmp/important.txt")
        except InterruptRejected as e:
            print(f"Operation rejected: {e.message}")
        ```

    Example (Modified Arguments):
        ```python
        try:
            delete_file("/tmp/test.txt")
        except HumanInTheLoopRequired as e:
            # User modifies the path
            modified_args = {"path": "/tmp/different.txt"}
            set_interrupt_approval(e.interrupt_id, approved=True, data=modified_args)

        # Retry - executes with modified arguments
        result = delete_file("/tmp/test.txt")  # Actually deletes /tmp/different.txt
        ```

    Args:
        func: The function to make interruptible

    Returns:
        Wrapped function that checks interrupt context
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        interrupt_id, all_kwargs = _generate_interrupt_id(func, args, kwargs)

        # Check approval status and get kwargs to use
        execution_kwargs = _handle_interrupt_approval(interrupt_id, func.__name__, all_kwargs)

        # Execute and cleanup
        result = await _execute_function_async(func, execution_kwargs)
        clear_interrupt(interrupt_id)
        return result

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        interrupt_id, all_kwargs = _generate_interrupt_id(func, args, kwargs)

        # Check approval status and get kwargs to use
        execution_kwargs = _handle_interrupt_approval(interrupt_id, func.__name__, all_kwargs)

        # Execute and cleanup
        result = func(**execution_kwargs)
        clear_interrupt(interrupt_id)
        return result

    if inspect.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore
