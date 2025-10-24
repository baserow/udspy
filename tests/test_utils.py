"""Tests for utility functions."""

import asyncio

import pytest

from udspy.utils import asyncify


def sync_function(x: int, y: int) -> int:
    """A sync function for testing."""
    return x + y


async def async_function(x: int, y: int) -> int:
    """An async function for testing."""
    await asyncio.sleep(0)
    return x + y


@pytest.mark.asyncio
async def test_asyncify_with_sync_function() -> None:
    """Test asyncify wraps sync functions."""
    wrapped = asyncify(sync_function)

    # Should be a coroutine function now
    assert asyncio.iscoroutinefunction(wrapped)

    # Should work correctly
    result = await wrapped(5, 3)
    assert result == 8


@pytest.mark.asyncio
async def test_asyncify_with_async_function() -> None:
    """Test asyncify returns async functions as-is."""
    wrapped = asyncify(async_function)

    # Should be the same function
    assert wrapped is async_function

    # Should still work
    result = await wrapped(5, 3)
    assert result == 8
