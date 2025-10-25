"""Tests for interrupt system."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from udspy.interrupt import (
    HumanInTheLoopRequired,
    InterruptRejected,
    ToolCall,
    clear_all_interrupts,
    clear_interrupt,
    get_interrupt_context,
    interruptible,
    set_interrupt_approval,
)


def test_interrupt_exception_attributes() -> None:
    """Test that HumanInTheLoopRequired exception has correct attributes."""
    tool_call = ToolCall(name="delete_file", args={"path": "/tmp/test.txt"})
    exc = HumanInTheLoopRequired(
        question="Confirm action?",
        tool_call=tool_call,
        context={"iteration": 1},
    )

    assert exc.question == "Confirm action?"
    assert exc.tool_call is not None
    assert exc.tool_call.name == "delete_file"
    assert exc.tool_call.args == {"path": "/tmp/test.txt"}
    assert exc.context == {"iteration": 1}
    assert exc.interrupt_id is not None
    assert isinstance(exc.interrupt_id, str)


def test_interrupt_context_management() -> None:
    """Test interrupt context get/set/clear operations."""
    clear_all_interrupts()

    # Initially empty
    ctx = get_interrupt_context()
    assert ctx == {}

    # Add an approval
    set_interrupt_approval("test-id-1", approved=True)
    ctx = get_interrupt_context()
    assert "test-id-1" in ctx
    assert ctx["test-id-1"]["approved"] is True

    # Add another with data
    set_interrupt_approval("test-id-2", approved=True, data={"key": "value"})
    ctx = get_interrupt_context()
    assert "test-id-2" in ctx
    assert ctx["test-id-2"]["data"] == {"key": "value"}

    # Clear one
    clear_interrupt("test-id-1")
    ctx = get_interrupt_context()
    assert "test-id-1" not in ctx
    assert "test-id-2" in ctx

    # Clear all
    clear_all_interrupts()
    ctx = get_interrupt_context()
    assert ctx == {}


def test_interruptible_decorator_raises_on_first_call() -> None:
    """Test that @interruptible raises HumanInTheLoopRequired on first call."""
    clear_all_interrupts()

    @interruptible
    def delete_file(path: str) -> str:
        return f"Deleted {path}"

    # First call should raise
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        delete_file("/tmp/test.txt")

    exc = exc_info.value
    assert "delete_file" in exc.question
    assert exc.tool_call is not None
    assert exc.tool_call.name == "delete_file"
    assert exc.tool_call.args == {"path": "/tmp/test.txt"}


def test_interruptible_decorator_proceeds_after_approval() -> None:
    """Test that @interruptible proceeds after approval."""
    clear_all_interrupts()

    call_count = {"count": 0}

    @interruptible
    def delete_file(path: str) -> str:
        call_count["count"] += 1
        return f"Deleted {path}"

    # First call raises
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        delete_file("/tmp/test.txt")

    # No execution yet
    assert call_count["count"] == 0

    # Approve the interrupt
    interrupt_id = exc_info.value.interrupt_id
    set_interrupt_approval(interrupt_id, approved=True)

    # Second call should succeed
    result = delete_file("/tmp/test.txt")
    assert result == "Deleted /tmp/test.txt"
    assert call_count["count"] == 1

    # Interrupt should be cleared after execution
    ctx = get_interrupt_context()
    assert interrupt_id not in ctx


def test_interruptible_with_modified_args() -> None:
    """Test @interruptible with modified arguments."""
    clear_all_interrupts()

    @interruptible
    def write_file(path: str, content: str) -> str:
        return f"Wrote '{content}' to {path}"

    # First call raises
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        write_file("/tmp/test.txt", "hello")

    # Approve with modified args
    interrupt_id = exc_info.value.interrupt_id
    modified_args = {"path": "/tmp/modified.txt", "content": "modified"}
    set_interrupt_approval(interrupt_id, approved=True, data=modified_args)

    # Second call should use modified args
    result = write_file("/tmp/test.txt", "hello")
    assert result == "Wrote 'modified' to /tmp/modified.txt"


@pytest.mark.asyncio
async def test_interruptible_async_function() -> None:
    """Test @interruptible with async function."""
    clear_all_interrupts()

    @interruptible
    async def async_delete(path: str) -> str:
        await asyncio.sleep(0.01)
        return f"Deleted {path}"

    # First call raises
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        await async_delete("/tmp/test.txt")

    # Approve
    interrupt_id = exc_info.value.interrupt_id
    set_interrupt_approval(interrupt_id, approved=True)

    # Second call succeeds
    result = await async_delete("/tmp/test.txt")
    assert result == "Deleted /tmp/test.txt"


def test_interruptible_thread_safety() -> None:
    """Test that interrupt context is thread-safe."""
    clear_all_interrupts()

    @interruptible
    def thread_func(thread_id: int) -> str:
        return f"Thread {thread_id}"

    results = []

    def worker(thread_id: int) -> None:
        try:
            # First call raises in each thread
            thread_func(thread_id)
        except HumanInTheLoopRequired as exc:
            # Approve in this thread
            set_interrupt_approval(exc.interrupt_id, approved=True)
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
async def test_interruptible_task_safety() -> None:
    """Test that interrupt context is asyncio task-safe."""
    clear_all_interrupts()

    @interruptible
    async def task_func(task_id: int) -> str:
        await asyncio.sleep(0.01)
        return f"Task {task_id}"

    results = []

    async def worker(task_id: int) -> None:
        try:
            await task_func(task_id)
        except HumanInTheLoopRequired as exc:
            set_interrupt_approval(exc.interrupt_id, approved=True)
            result = await task_func(task_id)
            results.append((task_id, result))

    # Run concurrent tasks
    await asyncio.gather(*[worker(i) for i in range(3)])

    # Each task should have executed successfully
    assert len(results) == 3
    task_ids = [r[0] for r in results]
    assert sorted(task_ids) == [0, 1, 2]


def test_interruptible_id_generation() -> None:
    """Test that interrupt IDs are consistent for same function and args."""
    clear_all_interrupts()

    @interruptible
    def func(x: int, y: str) -> str:
        return f"{x}-{y}"

    # Get first interrupt ID
    with pytest.raises(HumanInTheLoopRequired) as exc1:
        func(1, "a")
    id1 = exc1.value.interrupt_id

    # Same args should generate same ID
    with pytest.raises(HumanInTheLoopRequired) as exc2:
        func(1, "a")
    id2 = exc2.value.interrupt_id

    assert id1 == id2

    # Different args should generate different ID
    with pytest.raises(HumanInTheLoopRequired) as exc3:
        func(2, "b")
    id3 = exc3.value.interrupt_id

    assert id1 != id3


def test_interruptible_clears_after_execution() -> None:
    """Test that interrupt is cleared from context after successful execution."""
    clear_all_interrupts()

    @interruptible
    def func() -> str:
        return "done"

    # Raise and approve
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        func()
    interrupt_id = exc_info.value.interrupt_id
    set_interrupt_approval(interrupt_id, approved=True)

    # Execute
    result = func()
    assert result == "done"

    # Should be cleared
    ctx = get_interrupt_context()
    assert interrupt_id not in ctx


def test_interruptible_with_no_approval() -> None:
    """Test that @interruptible keeps raising without approval."""
    clear_all_interrupts()

    @interruptible
    def func() -> str:
        return "done"

    # Should raise every time without approval
    for _ in range(3):
        with pytest.raises(HumanInTheLoopRequired):
            func()


def test_interruptible_raises_rejected_when_user_rejects() -> None:
    """Test that @interruptible raises InterruptRejected when user explicitly rejects."""
    clear_all_interrupts()

    @interruptible
    def delete_file(path: str) -> str:
        return f"Deleted {path}"

    # First call - raises HumanInTheLoopRequired
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        delete_file("/tmp/test.txt")

    interrupt_id = exc_info.value.interrupt_id

    # User rejects the interrupt
    set_interrupt_approval(interrupt_id, approved=False, status="rejected")

    # Second call - should raise InterruptRejected
    with pytest.raises(InterruptRejected) as rejected_exc:
        delete_file("/tmp/test.txt")

    assert rejected_exc.value.interrupt_id == interrupt_id
    assert "rejected" in rejected_exc.value.message.lower()
    assert rejected_exc.value.tool_call is not None
    assert rejected_exc.value.tool_call.name == "delete_file"


@pytest.mark.asyncio
async def test_interruptible_async_raises_rejected() -> None:
    """Test that @interruptible async function raises InterruptRejected on rejection."""
    clear_all_interrupts()

    @interruptible
    async def async_delete(path: str) -> str:
        return f"Deleted {path}"

    # First call - raises HumanInTheLoopRequired
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        await async_delete("/tmp/test.txt")

    interrupt_id = exc_info.value.interrupt_id

    # User rejects
    set_interrupt_approval(interrupt_id, approved=False, status="rejected")

    # Second call - should raise InterruptRejected
    with pytest.raises(InterruptRejected) as rejected_exc:
        await async_delete("/tmp/test.txt")

    assert rejected_exc.value.interrupt_id == interrupt_id
    assert rejected_exc.value.tool_call is not None
