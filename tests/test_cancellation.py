"""Tests for asyncio cancellation support."""

import asyncio

import pytest

import udspy
from udspy import BaseCallback, InputField, OutputField, Predict, Prediction, Signature, settings
from udspy.streaming import OutputStreamChunk


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


class SlowQA(Signature):
    """Answer questions slowly for cancellation testing."""

    question: str = InputField()
    answer: str = OutputField()


def _make_slow_streaming_mock(delay: float = 0.05):
    """Create a slow streaming mock that yields chunks with delays."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    async def mock_create(**kwargs):
        async def mock_stream():
            parts = ['{"answer": "word1', " word2", " word3", " word4", ' word5"}']
            for i, part in enumerate(parts):
                await asyncio.sleep(delay)
                yield ChatCompletionChunk(
                    id="test",
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                    created=1234567890,
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(
                                content=part,
                                role="assistant" if i == 0 else None,
                            ),
                            finish_reason="stop" if i == len(parts) - 1 else None,
                        )
                    ],
                )

        return mock_stream()

    return mock_create


@pytest.mark.asyncio
async def test_astream_break_cancels_task():
    """Breaking out of astream() should cancel the internal LLM task."""
    settings.lm.client.chat.completions.create = _make_slow_streaming_mock(delay=0.05)

    predictor = Predict(SlowQA)
    events = []

    async for event in predictor.astream(question="Tell me something"):
        events.append(event)
        if isinstance(event, OutputStreamChunk) and len(events) >= 2:
            break  # Break early — should cancel the internal task

    # We got some events but not the full prediction
    assert len(events) >= 1
    assert not any(isinstance(e, Prediction) for e in events)

    # Give the event loop a tick to process the cancellation
    await asyncio.sleep(0.1)

    # Verify no lingering tasks (aside from the current test task)
    remaining = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
    for t in remaining:
        # None of the remaining tasks should be our LLM streaming task
        assert "aexecute" not in str(t.get_coro()), (
            "LLM task was not cancelled after breaking out of astream()"
        )


@pytest.mark.asyncio
async def test_aforward_cancel_propagates():
    """Cancelling an aforward() task should propagate CancelledError."""

    # Use a slow mock so we can cancel mid-flight
    async def slow_create(**kwargs):
        await asyncio.sleep(10)  # Long enough that we always cancel first

    settings.lm.client.chat.completions.create = slow_create

    predictor = Predict(QA)

    task = asyncio.create_task(predictor.aforward(question="Slow question"))

    # Let it start
    await asyncio.sleep(0.05)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_callback_receives_cancelled_error():
    """Callbacks should receive CancelledError as the exception on cancellation."""

    class CancellationRecorder(BaseCallback):
        def __init__(self):
            self.end_exception = None
            self.end_called = False

        def on_module_end(self, call_id, outputs, exception):
            self.end_called = True
            self.end_exception = exception

    callback = CancellationRecorder()

    async def slow_create(**kwargs):
        await asyncio.sleep(10)

    settings.lm.client.chat.completions.create = slow_create

    predictor = Predict(QA)

    with udspy.settings.context(callbacks=[callback]):
        task = asyncio.create_task(predictor.aforward(question="Will be cancelled"))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert callback.end_called, "on_module_end should have been called"
    assert isinstance(callback.end_exception, asyncio.CancelledError), (
        f"Expected CancelledError, got {type(callback.end_exception)}"
    )


@pytest.mark.asyncio
async def test_astream_full_consumption_no_cancel():
    """Fully consuming astream() should not trigger cancellation — regression check."""

    async def mock_create(**kwargs):
        async def mock_stream():
            from openai.types.chat import ChatCompletionChunk
            from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

            yield ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content='{"answer": "Paris"}', role="assistant"),
                        finish_reason="stop",
                    )
                ],
            )

        return mock_stream()

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA)
    events = []

    async for event in predictor.astream(question="Capital of France?"):
        events.append(event)

    # Should have completed normally with a Prediction
    predictions = [e for e in events if isinstance(e, Prediction)]
    assert len(predictions) == 1
    assert predictions[0].answer == "Paris"
