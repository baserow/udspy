"""Tests for base Module class error paths."""

import pytest

from udspy.module import Module
from udspy.streaming import Prediction


class TestModule(Module):
    """Test module for error path testing."""

    pass


@pytest.mark.asyncio
async def test_base_module_astream_not_implemented() -> None:
    """Test that base Module.astream() raises NotImplementedError."""
    module = TestModule()

    with pytest.raises(NotImplementedError, match="TestModule must implement astream"):
        async for _ in module.astream(input="test"):
            pass


@pytest.mark.asyncio
async def test_aforward_without_prediction() -> None:
    """Test aforward raises error when astream doesn't yield Prediction."""

    class BrokenModule(Module):
        async def astream(self, **inputs):  # type: ignore[override]
            yield "not a prediction"  # type: ignore[misc]

    module = BrokenModule()

    with pytest.raises(RuntimeError, match="did not yield a Prediction"):
        await module.aforward(input="test")


def test_forward_in_async_context() -> None:
    """Test forward() raises error when called from async context."""
    import asyncio

    class TestModuleWithAstream(Module):
        async def astream(self, **inputs):  # type: ignore[override]
            yield Prediction(answer="test")

    async def call_from_async() -> None:
        module = TestModuleWithAstream()
        # Should raise error when called from async context
        module.forward(input="test")

    with pytest.raises(RuntimeError, match="Cannot call.*from async context"):
        asyncio.run(call_from_async())
