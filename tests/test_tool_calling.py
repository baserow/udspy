"""Tests for tool calling functionality."""

from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, Field

from udspy import InputField, OutputField, Predict, Prediction, Signature, settings


class Calculator(BaseModel):
    """Perform arithmetic operations."""

    operation: str = Field(description="The operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class MathQuery(Signature):
    """Answer math questions using available tools."""

    question: str = InputField(description="Math question")
    answer: str = OutputField(description="Answer to the question")


@pytest.mark.asyncio
async def test_tool_calling_with_content() -> None:
    """Test that tool calls are captured and included in result."""

    # Create mock streaming response with tool calls
    chunks = [
        # First chunk with tool call start
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_123",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="Calculator",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        # Second chunk with arguments
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments='{"operation": "multiply", "a": 157, "b": 234}'
                                ),
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        ),
        # Third chunk with answer field marker
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\n"),
                    finish_reason=None,
                )
            ],
        ),
        # Fourth chunk with answer content
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="36738"),
                    finish_reason=None,
                )
            ],
        ),
    ]

    async def mock_stream() -> list[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = AsyncMock(return_value=mock_stream())

    # Create predictor with tools
    predictor = Predict(MathQuery, tools=[Calculator])

    # Test streaming
    events_received = []
    async for event in predictor.astream(question="What is 157 multiplied by 234?"):
        events_received.append(event)

    # Verify we got events
    assert len(events_received) > 0

    # Get final prediction
    result = events_received[-1]
    assert isinstance(result, Prediction)

    # Verify answer field
    assert result.answer == "36738"

    # Verify tool calls are present
    assert "tool_calls" in result
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call["id"] == "call_123"
    assert tool_call["name"] == "Calculator"
    assert tool_call["arguments"] == '{"operation": "multiply", "a": 157, "b": 234}'


@pytest.mark.asyncio
async def test_tool_calling_without_content() -> None:
    """Test tool calls when there's no content response (only tool calls)."""

    # Create mock streaming response with only tool calls, no content
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_456",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="Calculator",
                                    arguments='{"operation": "add", "a": 5, "b": 3}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
    ]

    async def mock_stream() -> list[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = AsyncMock(return_value=mock_stream())

    predictor = Predict(MathQuery, tools=[Calculator])

    # Get result
    result = await predictor.aforward(question="What is 5 plus 3?")

    # Verify tool calls are present even without content
    assert isinstance(result, Prediction)
    assert "tool_calls" in result
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call["id"] == "call_456"
    assert tool_call["name"] == "Calculator"
    assert tool_call["arguments"] == '{"operation": "add", "a": 5, "b": 3}'


@pytest.mark.asyncio
async def test_multiple_tool_calls() -> None:
    """Test handling multiple tool calls in the same response."""

    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="Calculator",
                                    arguments='{"operation": "add", "a": 5, "b": 3}',
                                ),
                            ),
                            ChoiceDeltaToolCall(
                                index=1,
                                id="call_2",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="Calculator",
                                    arguments='{"operation": "multiply", "a": 2, "b": 4}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
    ]

    async def mock_stream() -> list[ChatCompletionChunk]:
        for chunk in chunks:
            yield chunk

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = AsyncMock(return_value=mock_stream())

    predictor = Predict(MathQuery, tools=[Calculator])
    result = await predictor.aforward(question="Calculate something")

    # Verify both tool calls are present
    assert isinstance(result, Prediction)
    assert "tool_calls" in result
    assert len(result.tool_calls) == 2

    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["name"] == "Calculator"

    assert result.tool_calls[1]["id"] == "call_2"
    assert result.tool_calls[1]["name"] == "Calculator"
