"""Tests for streaming functionality."""

from unittest.mock import AsyncMock

import pytest
from conftest import make_mock_response
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from udspy import InputField, OutputField, Predict, Prediction, Signature
from udspy.streaming import OutputStreamChunk, StreamEvent, emit_event


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


@pytest.mark.asyncio
async def test_predict_astream() -> None:
    """Test async streaming with Predict.astream()."""

    # Create mock streaming response
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]", role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Paris", role=None),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="", role=None),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    from udspy import settings

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def mock_stream():
            for chunk in chunks:
                yield chunk

        return mock_stream()

    mock_async_client = settings.aclient
    mock_async_client.chat.completions.create = mock_create

    predictor = Predict(QA)
    events_received = []

    async for event in predictor.astream(question="What is the capital of France?"):
        events_received.append(event)

    assert len(events_received) > 0

    assert isinstance(events_received[-1], Prediction)

    chunks_received = [e for e in events_received if isinstance(e, OutputStreamChunk)]
    assert len(chunks_received) > 0


@pytest.mark.asyncio
async def test_predict_aforward() -> None:
    """Test async non-streaming with Predict.aforward()."""

    from udspy import settings

    mock_async_client = settings.aclient
    mock_response = make_mock_response("[[ ## answer ## ]]\nParis")
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

    predictor = Predict(QA)
    result = await predictor.aforward(question="What is the capital of France?")

    assert isinstance(result, Prediction)
    assert result.answer == "Paris"


def test_predict_forward_sync() -> None:
    """Test sync non-streaming with Predict.forward()."""

    from udspy import settings

    mock_async_client = settings.aclient
    mock_response = make_mock_response("[[ ## answer ## ]]\nParis")
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

    predictor = Predict(QA)
    result = predictor(question="What is the capital of France?")

    assert isinstance(result, Prediction)
    assert result.answer == "Paris"


@pytest.mark.asyncio
async def test_stream_chunk() -> None:
    """Test OutputStreamChunk creation."""
    predict = Predict(QA)
    chunk = OutputStreamChunk(
        predict, field_name="answer", delta=" is", content="Paris is", is_complete=False
    )

    assert chunk.module == predict
    assert chunk.field_name == "answer"
    assert chunk.delta == " is"
    assert chunk.content == "Paris is"
    assert not chunk.is_complete

    complete_chunk = OutputStreamChunk(
        predict, field_name="answer", delta="", content="Paris", is_complete=True
    )
    assert complete_chunk.is_complete


@pytest.mark.asyncio
async def test_emit_event() -> None:
    """Test emitting custom events to the stream."""

    class CustomStatus(StreamEvent):
        def __init__(self, message: str):
            self.message = message

    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]Paris", role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
    ]

    from udspy import settings

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def mock_stream():
            await emit_event(CustomStatus("Processing..."))
            for chunk in chunks:
                yield chunk

        return mock_stream()

    mock_async_client = settings.aclient
    mock_async_client.chat.completions.create = mock_create

    predictor = Predict(QA)
    events_received = []

    async for event in predictor.astream(question="Test?"):
        events_received.append(event)

    # Should have received custom event
    custom_events = [e for e in events_received if isinstance(e, CustomStatus)]
    assert len(custom_events) > 0
    assert custom_events[0].message == "Processing..."
