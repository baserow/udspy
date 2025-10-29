"""Tests for confirmation system."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from udspy.confirmation import (
    ConfirmationRejected,
    ConfirmationRequired,
    clear_all_confirmations,
    clear_confirmation,
    confirm_first,
    get_confirmation_context,
    respond_to_confirmation,
)
from udspy.tool import ToolCall


def test_confirmation_exception_attributes() -> None:
    """Test that ConfirmationRequired exception has correct attributes."""
    tool_call = ToolCall(name="delete_file", args={"path": "/tmp/test.txt"})
    exc = ConfirmationRequired(
        question="Confirm action?",
        tool_call=tool_call,
        context={"iteration": 1},
    )

    assert exc.question == "Confirm action?"
    assert exc.tool_call is not None
    assert exc.tool_call.name == "delete_file"
    assert exc.tool_call.args == {"path": "/tmp/test.txt"}
    assert exc.context == {"iteration": 1}
    assert exc.confirmation_id is not None
    assert isinstance(exc.confirmation_id, str)


def test_confirmation_context_management() -> None:
    """Test confirmation context get/set/clear operations."""
    clear_all_confirmations()

    # Initially empty
    ctx = get_confirmation_context()
    assert ctx == {}

    # Add an approval
    respond_to_confirmation("test-id-1", approved=True)
    ctx = get_confirmation_context()
    assert "test-id-1" in ctx
    assert ctx["test-id-1"]["approved"] is True

    # Add another with data
    respond_to_confirmation("test-id-2", approved=True, data={"key": "value"})
    ctx = get_confirmation_context()
    assert "test-id-2" in ctx
    assert ctx["test-id-2"]["data"] == {"key": "value"}

    # Clear one
    clear_confirmation("test-id-1")
    ctx = get_confirmation_context()
    assert "test-id-1" not in ctx
    assert "test-id-2" in ctx

    # Clear all
    clear_all_confirmations()
    ctx = get_confirmation_context()
    assert ctx == {}


def test_confirm_first_decorator_raises_on_first_call() -> None:
    """Test that @confirm_first raises ConfirmationRequired on first call."""
    clear_all_confirmations()

    @confirm_first
    def delete_file(path: str) -> str:
        return f"Deleted {path}"

    # First call should raise
    with pytest.raises(ConfirmationRequired) as exc_info:
        delete_file("/tmp/test.txt")

    exc = exc_info.value
    assert "delete_file" in exc.question
    assert exc.tool_call is not None
    assert exc.tool_call.name == "delete_file"
    assert exc.tool_call.args == {"path": "/tmp/test.txt"}


def test_confirm_first_decorator_proceeds_after_approval() -> None:
    """Test that @confirm_first proceeds after approval."""
    clear_all_confirmations()

    call_count = {"count": 0}

    @confirm_first
    def delete_file(path: str) -> str:
        call_count["count"] += 1
        return f"Deleted {path}"

    # First call raises
    with pytest.raises(ConfirmationRequired) as exc_info:
        delete_file("/tmp/test.txt")

    # No execution yet
    assert call_count["count"] == 0

    # Approve the confirmation
    confirmation_id = exc_info.value.confirmation_id
    respond_to_confirmation(confirmation_id, approved=True)

    # Second call should succeed
    result = delete_file("/tmp/test.txt")
    assert result == "Deleted /tmp/test.txt"
    assert call_count["count"] == 1

    # Confirmation should be cleared after execution
    ctx = get_confirmation_context()
    assert confirmation_id not in ctx


def test_confirm_first_with_modified_args() -> None:
    """Test @confirm_first with modified arguments."""
    clear_all_confirmations()

    @confirm_first
    def write_file(path: str, content: str) -> str:
        return f"Wrote '{content}' to {path}"

    # First call raises
    with pytest.raises(ConfirmationRequired) as exc_info:
        write_file("/tmp/test.txt", "hello")

    # Approve with modified args
    confirmation_id = exc_info.value.confirmation_id
    modified_args = {"path": "/tmp/modified.txt", "content": "modified"}
    respond_to_confirmation(confirmation_id, approved=True, data=modified_args)

    # Second call should use modified args
    result = write_file("/tmp/test.txt", "hello")
    assert result == "Wrote 'modified' to /tmp/modified.txt"


@pytest.mark.asyncio
async def test_confirm_first_async_function() -> None:
    """Test @confirm_first with async function."""
    clear_all_confirmations()

    @confirm_first
    async def async_delete(path: str) -> str:
        await asyncio.sleep(0.01)
        return f"Deleted {path}"

    # First call raises
    with pytest.raises(ConfirmationRequired) as exc_info:
        await async_delete("/tmp/test.txt")

    # Approve
    confirmation_id = exc_info.value.confirmation_id
    respond_to_confirmation(confirmation_id, approved=True)

    # Second call succeeds
    result = await async_delete("/tmp/test.txt")
    assert result == "Deleted /tmp/test.txt"


def test_confirm_first_thread_safety() -> None:
    """Test that confirmation context is thread-safe."""
    clear_all_confirmations()

    @confirm_first
    def thread_func(thread_id: int) -> str:
        return f"Thread {thread_id}"

    results = []

    def worker(thread_id: int) -> None:
        try:
            # First call raises in each thread
            thread_func(thread_id)
        except ConfirmationRequired as exc:
            # Approve in this thread
            respond_to_confirmation(exc.confirmation_id, approved=True)
            # Second call succeeds
            result = thread_func(thread_id)
            results.append((thread_id, result))

    # Run in multiple threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, i) for i in range(3)]
        for future in futures:
            future.result()

    # Each thread should have executed successfully
    assert len(results) == 3
    thread_ids = [r[0] for r in results]
    assert sorted(thread_ids) == [0, 1, 2]


@pytest.mark.asyncio
async def test_confirm_first_task_safety() -> None:
    """Test that confirmation context is asyncio task-safe."""
    clear_all_confirmations()

    @confirm_first
    async def task_func(task_id: int) -> str:
        await asyncio.sleep(0.01)
        return f"Task {task_id}"

    results = []

    async def worker(task_id: int) -> None:
        try:
            await task_func(task_id)
        except ConfirmationRequired as exc:
            respond_to_confirmation(exc.confirmation_id, approved=True)
            result = await task_func(task_id)
            results.append((task_id, result))

    # Run concurrent tasks
    await asyncio.gather(*[worker(i) for i in range(3)])

    # Each task should have executed successfully
    assert len(results) == 3
    task_ids = [r[0] for r in results]
    assert sorted(task_ids) == [0, 1, 2]


def test_confirm_first_id_generation() -> None:
    """Test that confirmation IDs are consistent for same function and args."""
    clear_all_confirmations()

    @confirm_first
    def func(x: int, y: str) -> str:
        return f"{x}-{y}"

    # Get first confirmation ID
    with pytest.raises(ConfirmationRequired) as exc1:
        func(1, "a")
    id1 = exc1.value.confirmation_id

    # Same args should generate same ID
    with pytest.raises(ConfirmationRequired) as exc2:
        func(1, "a")
    id2 = exc2.value.confirmation_id

    assert id1 == id2

    # Different args should generate different ID
    with pytest.raises(ConfirmationRequired) as exc3:
        func(2, "b")
    id3 = exc3.value.confirmation_id

    assert id1 != id3


def test_confirm_first_clears_after_execution() -> None:
    """Test that confirmation is cleared from context after successful execution."""
    clear_all_confirmations()

    @confirm_first
    def func() -> str:
        return "done"

    # Raise and approve
    with pytest.raises(ConfirmationRequired) as exc_info:
        func()
    confirmation_id = exc_info.value.confirmation_id
    respond_to_confirmation(confirmation_id, approved=True)

    # Execute
    result = func()
    assert result == "done"

    # Should be cleared
    ctx = get_confirmation_context()
    assert confirmation_id not in ctx


def test_confirm_first_with_no_approval() -> None:
    """Test that @confirm_first keeps raising without approval."""
    clear_all_confirmations()

    @confirm_first
    def func() -> str:
        return "done"

    # Should raise every time without approval
    for _ in range(3):
        with pytest.raises(ConfirmationRequired):
            func()


def test_confirm_first_raises_rejected_when_user_rejects() -> None:
    """Test that @confirm_first raises ConfirmationRejected when user explicitly rejects."""
    clear_all_confirmations()

    @confirm_first
    def delete_file(path: str) -> str:
        return f"Deleted {path}"

    # First call - raises ConfirmationRequired
    with pytest.raises(ConfirmationRequired) as exc_info:
        delete_file("/tmp/test.txt")

    confirmation_id = exc_info.value.confirmation_id

    # User rejects the confirmation
    respond_to_confirmation(confirmation_id, approved=False, status="rejected")

    # Second call - should raise ConfirmationRejected
    with pytest.raises(ConfirmationRejected) as rejected_exc:
        delete_file("/tmp/test.txt")

    assert rejected_exc.value.confirmation_id == confirmation_id
    assert "rejected" in rejected_exc.value.message.lower()
    assert rejected_exc.value.tool_call is not None
    assert rejected_exc.value.tool_call.name == "delete_file"


@pytest.mark.asyncio
async def test_confirm_first_async_raises_rejected() -> None:
    """Test that @confirm_first async function raises ConfirmationRejected on rejection."""
    clear_all_confirmations()

    @confirm_first
    async def async_delete(path: str) -> str:
        return f"Deleted {path}"

    # First call - raises ConfirmationRequired
    with pytest.raises(ConfirmationRequired) as exc_info:
        await async_delete("/tmp/test.txt")

    confirmation_id = exc_info.value.confirmation_id

    # User rejects
    respond_to_confirmation(confirmation_id, approved=False, status="rejected")

    # Second call - should raise ConfirmationRejected
    with pytest.raises(ConfirmationRejected) as rejected_exc:
        await async_delete("/tmp/test.txt")

    assert rejected_exc.value.confirmation_id == confirmation_id
    assert rejected_exc.value.tool_call is not None


@pytest.mark.asyncio
async def test_module_resume_state_parameter_passing() -> None:
    """Test that resume_state with ResumeState object is properly passed to aresume."""
    from unittest.mock import MagicMock

    from udspy import ResumeState
    from udspy.decorators import suspendable
    from udspy.module.base import Module

    # Create a test module
    class TestModule(Module):
        @suspendable
        async def aexecute(self, *, stream: bool = False, **inputs):
            # Not used in this test
            pass

        async def aresume(self, user_response: str, saved_state):
            # Track that resume was called with correct args
            return MagicMock(user_response=user_response, saved_state=saved_state)

    module = TestModule()

    # Create a mock saved state
    mock_exception = ConfirmationRequired(
        question="Test?", confirmation_id="test-id", tool_call=None, context={}
    )
    resume_state = ResumeState(mock_exception, "yes")

    # Test aforward delegates to aresume when resume_state is provided
    result = await module.aforward(resume_state=resume_state, other_input="value")

    assert result.user_response == "yes"
    assert result.saved_state == mock_exception


def test_module_resume_state_sync_forward() -> None:
    """Test that resume_state works with sync forward() method."""
    from unittest.mock import MagicMock

    from udspy import ResumeState
    from udspy.decorators import suspendable
    from udspy.module.base import Module

    # Create a test module
    class TestModule(Module):
        @suspendable
        async def aexecute(self, *, stream: bool = False, **inputs):
            # Not used in this test
            pass

        async def aresume(self, user_response: str, saved_state):
            # Track that resume was called
            return MagicMock(user_response=user_response, saved_state=saved_state)

    module = TestModule()

    # Create a mock saved state
    mock_exception = ConfirmationRequired(
        question="Test?", confirmation_id="test-id", tool_call=None, context={}
    )
    resume_state = ResumeState(mock_exception, "no")

    # Test forward delegates to aresume when resume_state is provided
    result = module.forward(resume_state=resume_state)

    assert result.user_response == "no"
    assert result.saved_state == mock_exception


def test_module_resume_state_call() -> None:
    """Test that resume_state works with __call__() method."""
    from unittest.mock import MagicMock

    from udspy import ResumeState
    from udspy.decorators import suspendable
    from udspy.module.base import Module

    # Create a test module
    class TestModule(Module):
        @suspendable
        async def aexecute(self, *, stream: bool = False, **inputs):
            # Not used in this test
            pass

        async def aresume(self, user_response: str, saved_state):
            # Track that resume was called
            return MagicMock(user_response=user_response, saved_state=saved_state)

    module = TestModule()

    # Create a mock saved state
    mock_exception = ConfirmationRequired(
        question="Test?", confirmation_id="test-id", tool_call=None, context={}
    )
    resume_state = ResumeState(mock_exception, "feedback: try again")

    # Test __call__ delegates to aresume when resume_state is provided
    result = module(resume_state=resume_state)

    assert result.user_response == "feedback: try again"
    assert result.saved_state == mock_exception


def test_module_resume_state_backward_compat_raw_exception() -> None:
    """Test backward compatibility with raw ConfirmationRequired exception."""
    from unittest.mock import MagicMock

    from udspy.decorators import suspendable
    from udspy.module.base import Module

    # Create a test module
    class TestModule(Module):
        @suspendable
        async def aexecute(self, *, stream: bool = False, **inputs):
            # Not used in this test
            pass

        async def aresume(self, user_response: str, saved_state):
            return MagicMock(user_response=user_response, saved_state=saved_state)

    module = TestModule()

    # Pass raw ConfirmationRequired as resume_state (backward compat)
    mock_exception = ConfirmationRequired(
        question="Test?", confirmation_id="test-id", tool_call=None, context={}
    )

    # When raw exception is passed, it should default to "yes"
    result = module.forward(resume_state=mock_exception)
    assert result.user_response == "yes"
    assert result.saved_state == mock_exception
