"""Tests for unified suspend/resume behavior across modules."""

import pytest

from udspy import InputField, Module, OutputField, Signature
from udspy.confirmation import ConfirmationRequired
from udspy.decorators import suspendable
from udspy.streaming import Prediction


class TestSignature(Signature):
    """Test signature."""

    question: str = InputField()
    answer: str = OutputField()


class SimpleTestModule(Module):
    """Test module that raises ConfirmationRequired."""

    def __init__(self):
        """Initialize without callbacks."""
        self.callbacks = []

    @suspendable
    async def aexecute(self, *, stream=False, **inputs):
        """Raise ConfirmationRequired for testing."""
        raise ConfirmationRequired(
            question="Proceed?",
            context={"test_data": "value"},
            tool_call=None,
            confirmation_id=None,
        )

    async def asuspend(self, exception):
        """Save state."""
        return {"saved": "state", "exception": exception}

    async def aresume(self, user_response, saved_state):
        """Resume from saved state."""
        return Prediction(answer=f"Resumed with: {user_response}")


@pytest.mark.asyncio
async def test_suspendable_decorator_catches_confirmation():
    """Test that @suspendable decorator catches ConfirmationRequired."""
    module = SimpleTestModule()

    with pytest.raises(ConfirmationRequired) as exc_info:
        await module.aforward(question="test")

    # Check that saved_state was attached to exception
    assert hasattr(exc_info.value, "saved_state")
    assert exc_info.value.saved_state["saved"] == "state"


@pytest.mark.asyncio
async def test_aforward_with_resume_state():
    """Test that aforward() can resume from a saved state."""
    module = SimpleTestModule()

    # First call raises ConfirmationRequired
    try:
        await module.aforward(question="test")
        raise AssertionError("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        saved_exception = e

    # Resume with user response
    from udspy.confirmation import ResumeState

    resume_state = ResumeState(saved_exception, "yes")
    result = await module.aforward(question="test", resume_state=resume_state)

    assert result.answer == "Resumed with: yes"


@pytest.mark.asyncio
async def test_astream_with_resume_state():
    """Test that astream() accepts resume_state parameter (integration pending)."""
    module = SimpleTestModule()

    # First call raises ConfirmationRequired
    try:
        async for _event in module.astream(question="test"):
            pass
        raise AssertionError("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        saved_exception = e

    # Resume with user response - verify no error (full integration TBD)
    from udspy.confirmation import ResumeState

    resume_state = ResumeState(saved_exception, "yes")

    # Just verify it doesn't crash - emit_prediction logic needs Module.astream update
    events = []
    async for event in module.astream(question="test", resume_state=resume_state):
        events.append(event)
        if isinstance(event, Prediction):
            break

    # At minimum, should not crash
    assert True  # If we got here, no exception was raised


def test_module_signatures_are_unified():
    """Test that all Module subclasses have consistent signatures."""
    # All should accept resume_state in aforward
    import inspect

    from udspy.module.predict import Predict
    from udspy.module.react import ReAct

    for module_class in [Predict, ReAct]:
        sig = inspect.signature(module_class.aforward)
        assert "resume_state" in sig.parameters, (
            f"{module_class.__name__}.aforward missing resume_state"
        )

        # All should have asuspend and aresume
        assert hasattr(module_class, "asuspend"), f"{module_class.__name__} missing asuspend"
        assert hasattr(module_class, "aresume"), f"{module_class.__name__} missing aresume"

        # All should have @suspendable on aexecute
        # (Check that aexecute has been wrapped)
        assert hasattr(module_class.aexecute, "__wrapped__"), (
            f"{module_class.__name__}.aexecute missing @suspendable decorator"
        )
