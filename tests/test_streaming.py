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

    # Create mock streaming response (JSON format)
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content='{"answer": "', role="assistant"),
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
                    delta=ChoiceDelta(content='"}', role=None),
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

    mock_async_client = settings.lm.client
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

    mock_async_client = settings.lm.client
    mock_response = make_mock_response('{"answer": "Paris"}')
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

    predictor = Predict(QA)
    result = await predictor.aforward(question="What is the capital of France?")

    assert isinstance(result, Prediction)
    assert result.answer == "Paris"


def test_predict_forward_sync() -> None:
    """Test sync non-streaming with Predict.forward()."""

    from udspy import settings

    mock_async_client = settings.lm.client
    mock_response = make_mock_response('{"answer": "Paris"}')
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
                    delta=ChoiceDelta(content='{"answer": "Paris"}', role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
    ]

    from udspy import settings

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def mock_stream():
            emit_event(CustomStatus("Processing..."))
            for chunk in chunks:
                yield chunk

        return mock_stream()

    mock_async_client = settings.lm.client
    mock_async_client.chat.completions.create = mock_create

    predictor = Predict(QA)
    events_received = []

    async for event in predictor.astream(question="Test?"):
        events_received.append(event)

    # Should have received custom event
    custom_events = [e for e in events_received if isinstance(e, CustomStatus)]
    assert len(custom_events) > 0
    assert custom_events[0].message == "Processing..."


@pytest.mark.asyncio
async def test_output_stream_chunk_completion_lifecycle() -> None:
    """Test that OutputStreamChunks are emitted with is_complete=False, then is_complete=True exactly once per field.

    This test verifies:
    1. Multiple incomplete chunks (is_complete=False) are emitted as content streams in
    2. Exactly one complete chunk (is_complete=True) is emitted per field when streaming finishes
    3. For multiple fields, each gets its own complete chunk
    """

    class MultiFieldSignature(Signature):
        """Multi-field output signature."""

        question: str = InputField()
        reasoning: str = OutputField()
        answer: str = OutputField()

    # Create mock streaming response with incremental JSON for multiple fields
    chunks = [
        # Start of JSON object
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content='{"reasoning": "Let me', role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        # Continue reasoning field
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=" think", role=None),
                    finish_reason=None,
                )
            ],
        ),
        # More reasoning
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=' step by step"', role=None),
                    finish_reason=None,
                )
            ],
        ),
        # Start answer field
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=', "answer": "The', role=None),
                    finish_reason=None,
                )
            ],
        ),
        # Continue answer
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=" answer is", role=None),
                    finish_reason=None,
                )
            ],
        ),
        # Finish answer
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content=' 42"}', role=None),
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

    mock_async_client = settings.lm.client
    mock_async_client.chat.completions.create = mock_create

    predictor = Predict(MultiFieldSignature)
    events_received = []

    async for event in predictor.astream(question="What is the meaning of life?"):
        events_received.append(event)

    # Separate events by type
    output_chunks = [e for e in events_received if isinstance(e, OutputStreamChunk)]
    prediction = [e for e in events_received if isinstance(e, Prediction)]

    # Should have received OutputStreamChunks
    assert len(output_chunks) > 0, "Should have received OutputStreamChunks"

    # Should have exactly one final Prediction
    assert len(prediction) == 1, "Should have exactly one Prediction"

    # Group chunks by field
    chunks_by_field = {}
    for chunk in output_chunks:
        if chunk.field_name not in chunks_by_field:
            chunks_by_field[chunk.field_name] = []
        chunks_by_field[chunk.field_name].append(chunk)

    # Verify both fields received chunks
    assert "reasoning" in chunks_by_field, "Should have chunks for 'reasoning' field"
    assert "answer" in chunks_by_field, "Should have chunks for 'answer' field"

    # For each field, verify the lifecycle
    for field_name, field_chunks in chunks_by_field.items():
        # Should have at least one chunk
        assert len(field_chunks) > 0, f"Field {field_name} should have at least one chunk"

        # All chunks except the last should have is_complete=False
        incomplete_chunks = [c for c in field_chunks if not c.is_complete]
        complete_chunks = [c for c in field_chunks if c.is_complete]

        # Should have at least one incomplete chunk (streaming in progress)
        assert len(incomplete_chunks) >= 1, (
            f"Field {field_name} should have incomplete chunks while streaming"
        )

        # Should have exactly one complete chunk
        assert len(complete_chunks) == 1, (
            f"Field {field_name} should have exactly one complete chunk"
        )

        # The complete chunk should be the last one
        assert field_chunks[-1].is_complete, f"Field {field_name}: last chunk should be complete"

        # All incomplete chunks should come before the complete chunk
        for i, chunk in enumerate(field_chunks[:-1]):
            assert not chunk.is_complete, f"Field {field_name}: chunk {i} should be incomplete"

        # Verify content accumulation
        # The complete chunk should have the full content
        complete_chunk = complete_chunks[0]
        if field_name == "reasoning":
            assert complete_chunk.content == "Let me think step by step", (
                f"Field {field_name} complete content mismatch"
            )
        elif field_name == "answer":
            assert complete_chunk.content == "The answer is 42", (
                f"Field {field_name} complete content mismatch"
            )


@pytest.mark.asyncio
async def test_sync_tool_can_emit_events() -> None:
    """Test that synchronous tools can emit events via context propagation.

    This verifies that execute_function_async properly copies the context when
    running sync functions in the executor, allowing sync tools to emit events
    that are received by the stream consumer.
    """
    import asyncio
    from dataclasses import dataclass

    from udspy.streaming import _stream_queue, emit_event
    from udspy.utils.async_support import execute_function_async

    @dataclass
    class ToolProgress(StreamEvent):
        message: str
        step: int

    def sync_function(count: int) -> str:
        """Synchronous function that emits progress events."""
        for i in range(count):
            # This is a sync function emitting events - should work via context propagation
            emit_event(ToolProgress(f"Step {i + 1}/{count}", i + 1))
        return f"Completed {count} steps"

    # Set up a stream queue (simulating Module.astream())
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
    token = _stream_queue.set(queue)

    try:
        # Call the sync function through execute_function_async (like tools do)
        result = await execute_function_async(sync_function, {"count": 3})

        # Verify the result
        assert result == "Completed 3 steps"

        # Collect all events from the queue
        events_received = []
        while not queue.empty():
            event = queue.get_nowait()
            if event is not None:
                events_received.append(event)

        # Verify we received progress events from the SYNCHRONOUS function
        progress_events = [e for e in events_received if isinstance(e, ToolProgress)]
        assert len(progress_events) == 3, f"Expected 3 progress events, got {len(progress_events)}"
        assert progress_events[0].message == "Step 1/3"
        assert progress_events[0].step == 1
        assert progress_events[1].message == "Step 2/3"
        assert progress_events[1].step == 2
        assert progress_events[2].message == "Step 3/3"
        assert progress_events[2].step == 3

    finally:
        _stream_queue.reset(token)
