"""Tests for streaming edge cases."""

import pytest

from udspy.streaming import StreamEvent, emit_event


class CustomEvent(StreamEvent):
    """Custom event for testing."""

    def __init__(self, message: str):
        self.message = message


@pytest.mark.asyncio
async def test_emit_event_without_queue() -> None:
    """Test emit_event silently ignores when no queue is set."""
    event = CustomEvent("test")

    # Should not raise, just silently ignore
    await emit_event(event)


@pytest.mark.asyncio
async def test_emit_event_with_none_queue() -> None:
    """Test emit_event handles None queue gracefully."""
    from udspy.streaming import _stream_queue

    # Set queue to None explicitly
    token = _stream_queue.set(None)

    try:
        event = CustomEvent("test")
        # Should not raise
        await emit_event(event)
    finally:
        _stream_queue.reset(token)
