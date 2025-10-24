"""Tests for streaming functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from udspy import InputField, OutputField, Predict, Signature, StreamingPredict, streamify
from udspy.streaming import StreamChunk


@pytest.mark.asyncio
async def test_streaming_predict() -> None:
    """Test basic streaming prediction."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

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

    # Create async iterator mock
    async def mock_stream():
        for chunk in chunks:
            yield chunk

    # Import settings here to use the configured mock client
    from udspy import settings

    mock_stream_obj = MagicMock()
    mock_stream_obj.__aiter__ = mock_stream
    mock_stream_obj.__aenter__ = AsyncMock(return_value=mock_stream())
    mock_stream_obj.__aexit__ = AsyncMock(return_value=None)

    mock_async_client = settings.async_client
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    predictor = StreamingPredict(QA)
    chunks_received = []

    async for item in predictor.stream(question="What is the capital of France?"):
        chunks_received.append(item)

    # Should receive stream chunks and final prediction
    assert len(chunks_received) > 0


@pytest.mark.asyncio
async def test_stream_chunk() -> None:
    """Test StreamChunk creation."""
    chunk = StreamChunk("answer", "Paris", is_complete=False)

    assert chunk.field_name == "answer"
    assert chunk.content == "Paris"
    assert not chunk.is_complete

    complete_chunk = StreamChunk("answer", "", is_complete=True)
    assert complete_chunk.is_complete


def test_streamify() -> None:
    """Test converting Predict to StreamingPredict."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    predictor = Predict(QA, temperature=0.5)
    streaming_predictor = streamify(predictor)

    assert isinstance(streaming_predictor, StreamingPredict)
    assert streaming_predictor.signature == QA
    assert streaming_predictor.model == predictor.model
    assert streaming_predictor.kwargs["temperature"] == 0.5
