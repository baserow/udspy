"""Tests for utility functions."""

import pytest

from udspy.utils import ensure_sync_context


def test_ensure_sync_context_allows_sync() -> None:
    """Test that ensure_sync_context allows execution in sync context."""
    # Should not raise - we're in sync context
    ensure_sync_context("TestClass.method")


@pytest.mark.asyncio
async def test_ensure_sync_context_blocks_async() -> None:
    """Test that ensure_sync_context raises error in async context."""
    with pytest.raises(RuntimeError) as exc_info:
        ensure_sync_context("TestClass.method")

    assert "Cannot call TestClass.method() from async context" in str(exc_info.value)
    assert "await testClass.amethod(...)" in str(exc_info.value)
