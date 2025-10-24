"""Tests for @tool decorator and automatic tool execution."""

import pytest
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


def test_tool_decorator() -> None:
    """Test @tool decorator creates Tool object."""
    assert calculator.name == "Calculator"
    assert calculator.description == "Perform arithmetic operations"
    assert "operation" in calculator.parameters
    assert "a" in calculator.parameters
    assert "b" in calculator.parameters


def test_tool_to_openai_schema() -> None:
    """Test Tool converts to OpenAI schema."""
    schema = calculator.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "Calculator"
    assert schema["function"]["description"] == "Perform arithmetic operations"

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "operation" in params["properties"]
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert set(params["required"]) == {"operation", "a", "b"}


def test_tool_callable() -> None:
    """Test Tool is callable."""
    result = calculator(operation="multiply", a=5, b=3)
    assert result == 15


@pytest.mark.asyncio
async def test_predict_with_tool_automatic_execution() -> None:
    """Test Predict automatically executes tools and handles multi-turn."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

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

    # Mock second streaming response - LLM provides final answer after seeing tool result
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
            # First call - return tool call
            async def first_stream():  # type: ignore[no-untyped-def]
                for chunk in first_chunks:
                    yield chunk

            return first_stream()
        else:
            # Second call - return final answer
            async def second_stream():  # type: ignore[no-untyped-def]
                for chunk in second_chunks:
                    yield chunk

            return second_stream()

    # Mock the client
    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    # Create predictor with tool
    predictor = Predict(QA, tools=[calculator])

    # Call predictor - should automatically handle tool execution
    result = await predictor.aforward(question="What is 5 times 3?")

    # Verify the result
    assert result.answer == "The answer is 15"

    # Verify two API calls were made
    assert call_count == 2


def test_predict_stores_tool_callables() -> None:
    """Test Predict stores tool callables correctly."""
    predictor = Predict(QA, tools=[calculator])

    assert "Calculator" in predictor.tool_callables
    assert predictor.tool_callables["Calculator"] == calculator
    assert calculator in predictor.tool_schemas


def test_tool_with_optional_types() -> None:
    """Test Tool handles Optional types correctly."""

    from pydantic import Field

    @tool(name="OptionalTool", description="Tool with optional params")
    def optional_tool(
        required: str = Field(description="Required param"),
        optional: str | None = Field(default=None, description="Optional param"),
    ) -> str:
        return f"{required}-{optional}"

    schema = optional_tool.to_openai_schema()

    # Check that types are converted correctly
    params = schema["function"]["parameters"]
    assert "required" in params["properties"]
    assert "optional" in params["properties"]

    # Required should be in required list
    assert "required" in params["required"]


@pytest.mark.asyncio
async def test_tool_error_handling() -> None:
    """Test error handling when tool execution fails."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    @tool(name="FailingTool", description="A tool that fails")
    def failing_tool(value: int = Field(description="A value")) -> int:
        raise ValueError("Tool failed!")

    # Mock first streaming response - LLM requests failing tool
    first_chunks = [
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
                                    name="FailingTool",
                                    arguments='{"value": 42}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    # Mock second streaming response - LLM provides answer after seeing error
    second_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\nThe tool encountered an error"),
                    finish_reason=None,
                )
            ],
        )
    ]

    call_count = 0
    messages_log = []

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        messages_log.append(kwargs.get("messages", []))

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

    predictor = Predict(QA, tools=[failing_tool])

    # Should handle the error gracefully
    result = await predictor.aforward(question="Test")

    # Verify it got an answer even though tool failed
    assert "error" in result.answer.lower()

    # Verify second call included error in tool message
    second_call_messages = messages_log[1]
    tool_message = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "Error executing tool" in tool_message["content"]
