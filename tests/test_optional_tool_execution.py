"""Tests for optional tool execution."""

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import Field

from udspy import InputField, OutputField, Predict, Signature, settings, tool


@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Test calculator tool."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


@pytest.mark.asyncio
async def test_auto_execute_tools_true() -> None:
    """Test with auto_execute_tools=True (default) - should execute automatically."""
    # Mock first streaming response - LLM requests tool
    first_chunks = [
        ChatCompletionChunk(
            id="test1",
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
                                    arguments='{"operation": "multiply", "a": 5, "b": 3}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    # Mock second streaming response - LLM provides final answer
    second_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="[[ ## answer ## ]]"), finish_reason=None)
            ],
        ),
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="\nThe answer is 15"), finish_reason=None)
            ],
        ),
    ]

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:

            async def first_stream():  # type: ignore[no-untyped-def]
                for chunk in first_chunks:
                    yield chunk

            return first_stream()
        else:

            async def second_stream():  # type: ignore[no-untyped-def]
                for chunk in second_chunks:
                    yield chunk

            return second_stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Should automatically execute tool and return final answer
    result = await predictor.aforward(question="What is 5 times 3?")

    assert result.answer == "The answer is 15"
    assert call_count == 2  # Two calls: initial + after tool execution


@pytest.mark.asyncio
async def test_auto_execute_tools_false() -> None:
    """Test with auto_execute_tools=False - should return tool_calls without execution."""
    # Mock streaming response - LLM requests tool
    chunks = [
        ChatCompletionChunk(
            id="test1",
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
                                    arguments='{"operation": "add", "a": 10, "b": 20}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        async def stream():  # type: ignore[no-untyped-def]
            for chunk in chunks:
                yield chunk

        return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Should NOT execute tool, just return Prediction with tool_calls
    result = await predictor.aforward(auto_execute_tools=False, question="What is 10 plus 20?")

    # Should have tool_calls in the result
    assert "tool_calls" in result
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "Calculator"
    assert result.tool_calls[0]["id"] == "call_123"
    assert '"operation": "add"' in result.tool_calls[0]["arguments"]

    # Should only make one call (no automatic execution)
    assert call_count == 1


def test_forward_with_auto_execute_tools_false() -> None:
    """Test sync forward() with auto_execute_tools=False."""
    # Mock streaming response - LLM requests tool
    chunks = [
        ChatCompletionChunk(
            id="test1",
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
                                    arguments='{"operation": "subtract", "a": 50, "b": 25}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        async def stream():  # type: ignore[no-untyped-def]
            for chunk in chunks:
                yield chunk

        return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Sync call with auto_execute_tools=False
    result = predictor.forward(auto_execute_tools=False, question="What is 50 minus 25?")

    # Should have tool_calls in the result
    assert "tool_calls" in result
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "Calculator"

    # Should only make one call
    assert call_count == 1


def test_call_with_auto_execute_tools_false() -> None:
    """Test __call__() with auto_execute_tools=False."""
    # Mock streaming response - LLM requests tool
    chunks = [
        ChatCompletionChunk(
            id="test1",
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
                                id="call_789",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="Calculator",
                                    arguments='{"operation": "divide", "a": 100, "b": 5}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            for chunk in chunks:
                yield chunk

        return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Call with auto_execute_tools=False
    result = predictor(auto_execute_tools=False, question="What is 100 divided by 5?")

    # Should have tool_calls in the result
    assert "tool_calls" in result
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "Calculator"
