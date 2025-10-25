"""Confirmation system for human-in-the-loop interactions."""

import functools
import inspect
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

# Thread-safe and asyncio task-safe confirmation context
_confirmation_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "confirmation_context", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


class ToolCall:
    """Information about a tool call that triggered a confirmation.

    This is optional - confirmations can occur without tool calls.
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


class ResumeState:
    """Container for resuming execution after a confirmation.

    This combines the original ConfirmationRequired exception with the user's response,
    providing a single object to pass to resume execution.

    Example:
        ```python
        try:
            result = agent(question="Delete files")
        except ConfirmationRequired as e:
            # Create resume state with user response
            resume_state = ResumeState(e, user_response="yes")
            result = agent(question="Delete files", resume_state=resume_state)
        ```
    """

    def __init__(self, exception: "ConfirmationRequired", user_response: str = "yes"):
        """Initialize resume state.

        Args:
            exception: The original ConfirmationRequired exception
            user_response: The user's response (default: "yes")
        """
        self.exception = exception
        self.user_response = user_response

    @property
    def confirmation_id(self) -> str:
        """Get confirmation ID from the exception."""
        return self.exception.confirmation_id

    @property
    def question(self) -> str:
        """Get question from the exception."""
        return self.exception.question

    @property
    def tool_call(self) -> "ToolCall | None":
        """Get tool call from the exception."""
        return self.exception.tool_call

    @property
    def context(self) -> dict[str, Any]:
        """Get context from the exception."""
        return self.exception.context


class ConfirmationRequired(Exception):
    """Raised when human input is needed to proceed.

    This exception pauses execution and allows modules to save state for resumption.
    It can be raised by:
    - Tools decorated with @confirm_first
    - Modules that need user input (e.g., ask_to_user)
    - Custom code requiring human interaction

    Attributes:
        question: The question being asked to the user
        confirmation_id: Unique ID for this confirmation request
        tool_call: Optional ToolCall information if raised by a tool
        context: General-purpose context dictionary for module state
    """

    def __init__(
        self,
        question: str,
        *,
        confirmation_id: str | None = None,
        tool_call: ToolCall | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize ConfirmationRequired exception.

        Args:
            question: Question to ask the user
            confirmation_id: Unique ID for this confirmation (auto-generated if not provided)
            tool_call: Optional tool call information
            context: Optional context dictionary for module-specific state
        """
        super().__init__(question)
        self.question = question
        self.confirmation_id = confirmation_id or str(uuid.uuid4())
        self.tool_call = tool_call
        self.context = context or {}


class ConfirmationRejected(Exception):
    """Raised when user rejects a confirmation/tool execution.

    This signals that the user explicitly does not want the operation to proceed,
    as opposed to just pending approval. This allows calling code to distinguish
    between "waiting for approval" and "user said no".

    Example:
        ```python
        from udspy import confirm_first, ConfirmationRequired, ConfirmationRejected
        from udspy import respond_to_confirmation

        @confirm_first
        def delete_database() -> str:
            return "Database deleted"

        try:
            delete_database()
        except ConfirmationRequired as e:
            # User rejects the dangerous operation
            respond_to_confirmation(e.confirmation_id, approved=False, status="rejected")

        try:
            delete_database()  # Try again
        except ConfirmationRejected as e:
            print(f"Operation stopped: {e.message}")
            # Can handle rejection differently than pending approval
        ```

    Attributes:
        message: Description of what was rejected
        confirmation_id: The confirmation ID that was rejected
        tool_call: Optional ToolCall information if raised by a tool
    """

    def __init__(
        self,
        message: str,
        *,
        confirmation_id: str | None = None,
        tool_call: ToolCall | None = None,
    ):
        """Initialize ConfirmationRejected exception.

        Args:
            message: Description of what was rejected
            confirmation_id: The confirmation ID that was rejected
            tool_call: Optional tool call information
        """
        super().__init__(message)
        self.message = message
        self.confirmation_id = confirmation_id
        self.tool_call = tool_call


def get_confirmation_context() -> dict[str, Any]:
    """Get the current confirmation context.

    Returns:
        Dictionary mapping confirmation IDs to their approval status/data
    """
    ctx = _confirmation_context.get()
    return ctx.copy() if ctx is not None else {}


def respond_to_confirmation(
    confirmation_id: str,
    approved: bool = True,
    data: Any = None,
    status: str | None = None,
) -> None:
    """Mark an confirmation as approved in the context.

    Args:
        confirmation_id: The confirmation ID to approve
        approved: Whether the confirmation is approved
        data: Optional data associated with the approval (e.g., modified args)
        status: Optional status string (e.g., "approved", "rejected", "edited", "feedback")
    """
    ctx = _confirmation_context.get()
    if ctx is None:
        ctx = {}
    else:
        ctx = ctx.copy()
    ctx[confirmation_id] = {
        "approved": approved,
        "data": data,
        "status": status or ("approved" if approved else "rejected"),
    }
    _confirmation_context.set(ctx)


def get_confirmation_status(confirmation_id: str) -> str | None:
    """Get the status of a confirmation.

    Args:
        confirmation_id: The confirmation ID to check

    Returns:
        Status string or None if confirmation not found.
        Possible values: "pending", "approved", "rejected", "edited", "feedback"
    """
    ctx = _confirmation_context.get()
    if ctx is None or confirmation_id not in ctx:
        return "pending"
    return ctx[confirmation_id].get(
        "status", "approved" if ctx[confirmation_id].get("approved") else "rejected"
    )


def clear_confirmation(confirmation_id: str) -> None:
    """Remove an confirmation from the context.

    Args:
        confirmation_id: The confirmation ID to clear
    """
    ctx = _confirmation_context.get()
    if ctx is None:
        return
    ctx = ctx.copy()
    ctx.pop(confirmation_id, None)
    _confirmation_context.set(ctx)


def clear_all_confirmations() -> None:
    """Clear all confirmations from the context."""
    _confirmation_context.set({})


def _generate_confirmation_id(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Generate a stable confirmation ID from function name and arguments.

    Creates a unique identifier for a confirmation based on the function being called
    and its arguments. This allows the same function call to be resumed after confirmation.

    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tuple of (confirmation_id, normalized_kwargs) where:
        - confirmation_id: Stable ID in format "function_name:hash(args)"
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
    confirmation_id = f"{func_name}:{hash(args_repr)}"

    return confirmation_id, all_kwargs


def _handle_confirmation_approval(
    confirmation_id: str, func_name: str, all_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Check confirmation approval status and handle accordingly.

    This function checks if a confirmation has been approved, rejected, or is pending.
    Based on the status, it either:
    - Returns modified kwargs if approved (possibly with user edits)
    - Raises ConfirmationRejected if user explicitly rejected
    - Raises ConfirmationRequired if pending approval

    Args:
        confirmation_id: The confirmation ID to check
        func_name: Name of the function being called
        all_kwargs: Normalized keyword arguments for the function

    Returns:
        Modified kwargs to use for execution (if approved)

    Raises:
        ConfirmationRejected: If user rejected the confirmation
        ConfirmationRequired: If confirmation is pending approval
    """
    import json

    ctx = _confirmation_context.get()
    approval = ctx.get(confirmation_id) if ctx is not None else None

    if approval:
        is_approved = approval.get("approved", False)
        status = approval.get("status", "pending")

        if is_approved:
            # Use modified args if provided, otherwise original
            return approval.get("data") or all_kwargs
        elif status == "rejected":
            # User explicitly rejected
            raise ConfirmationRejected(
                message=f"User rejected execution of {func_name}",
                confirmation_id=confirmation_id,
                tool_call=ToolCall(name=func_name, args=all_kwargs),
            )

    # No approval yet - ask user
    raise ConfirmationRequired(
        question=f"Confirm execution of {func_name} with args: {json.dumps(all_kwargs)}? (yes/no/feedback/edit)",
        confirmation_id=confirmation_id,
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


def confirm_first(func: F) -> F:
    """Decorator that requires confirmation before function execution.

    When called, the decorator checks the confirmation context and behaves as follows:
    1. **Pending** (no approval): Raises ConfirmationRequired
    2. **Approved**: Proceeds with execution (possibly with modified args)
    3. **Rejected**: Raises ConfirmationRejected

    The confirmation ID is generated based on function name and arguments to ensure
    the same call can be resumed after approval.

    Example (Approval Flow):
        ```python
        from udspy import confirm_first, ConfirmationRequired, respond_to_confirmation

        @confirm_first
        def delete_file(path: str) -> str:
            os.remove(path)
            return f"Deleted {path}"

        # First call - pending, raises ConfirmationRequired
        try:
            delete_file("/tmp/test.txt")
        except ConfirmationRequired as e:
            print(f"Confirm: {e.question}")
            # User approves
            respond_to_confirmation(e.confirmation_id, approved=True)

        # Retry - approved, executes normally
        result = delete_file("/tmp/test.txt")
        print(result)  # "Deleted /tmp/test.txt"
        ```

    Example (Rejection Flow):
        ```python
        from udspy import ConfirmationRejected

        try:
            delete_file("/tmp/important.txt")
        except ConfirmationRequired as e:
            # User rejects
            respond_to_confirmation(e.confirmation_id, approved=False, status="rejected")

        # Retry - rejected, raises ConfirmationRejected
        try:
            delete_file("/tmp/important.txt")
        except ConfirmationRejected as e:
            print(f"Operation rejected: {e.message}")
        ```

    Example (Modified Arguments):
        ```python
        try:
            delete_file("/tmp/test.txt")
        except ConfirmationRequired as e:
            # User modifies the path
            modified_args = {"path": "/tmp/different.txt"}
            respond_to_confirmation(e.confirmation_id, approved=True, data=modified_args)

        # Retry - executes with modified arguments
        result = delete_file("/tmp/test.txt")  # Actually deletes /tmp/different.txt
        ```

    Args:
        func: The function to require confirmation for

    Returns:
        Wrapped function that checks confirmation context
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        confirmation_id, all_kwargs = _generate_confirmation_id(func, args, kwargs)

        # Check approval status and get kwargs to use
        execution_kwargs = _handle_confirmation_approval(confirmation_id, func.__name__, all_kwargs)

        # Execute and cleanup
        result = await _execute_function_async(func, execution_kwargs)
        clear_confirmation(confirmation_id)
        return result

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        confirmation_id, all_kwargs = _generate_confirmation_id(func, args, kwargs)

        # Check approval status and get kwargs to use
        execution_kwargs = _handle_confirmation_approval(confirmation_id, func.__name__, all_kwargs)

        # Execute and cleanup
        result = func(**execution_kwargs)
        clear_confirmation(confirmation_id)
        return result

    if inspect.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore
