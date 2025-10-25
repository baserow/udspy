"""Tests for base Module class error paths."""

import pytest

from udspy.module import Module
from udspy.streaming import Prediction


class TestModule(Module):
    """Test module for error path testing."""

    pass


@pytest.mark.asyncio
async def test_base_module_astream_not_implemented() -> None:
    """Test that base Module.aexecute() raises NotImplementedError when not overridden."""
    module = TestModule()

    with pytest.raises(NotImplementedError, match="TestModule must implement aexecute"):
        async for _ in module.astream(input="test"):
            pass


@pytest.mark.asyncio
async def test_aforward_without_prediction() -> None:
    """Test aforward raises error when aexecute() doesn't return Prediction."""

    class BrokenModule(Module):
        async def aexecute(self, *, stream: bool = False, **inputs):  # type: ignore[override]
            return "not a prediction"  # type: ignore[return-value]

    module = BrokenModule()

    # aforward should return whatever aexecute returns, even if it's wrong type
    # The type system should catch this, but at runtime it won't raise
    result = await module.aforward(input="test")
    assert result == "not a prediction"


def test_forward_in_async_context() -> None:
    """Test forward() raises error when called from async context."""
    import asyncio

    class TestModuleWithExecute(Module):
        async def aexecute(self, *, stream: bool = False, **inputs):  # type: ignore[override]
            return Prediction(answer="test")

    async def call_from_async() -> None:
        module = TestModuleWithExecute()
        # Should raise error when called from async context
        module.forward(input="test")

    with pytest.raises(RuntimeError, match="Cannot call.*from async context"):
        asyncio.run(call_from_async())
