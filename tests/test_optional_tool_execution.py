"""Tests for optional tool execution."""

import pytest
from conftest import make_mock_response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
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
    # Mock first response - LLM requests tool
    first_response = ChatCompletion(
        id="test1",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="Calculator",
                                arguments='{"operation": "multiply", "a": 5, "b": 3}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response - LLM provides final answer
    second_response = make_mock_response('{"answer": "The answer is 15"}')

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return first_response
        else:
            return second_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])
    print("predictor:", predictor)
    # Should automatically execute tool and return final answer
    result = await predictor.aforward(question="What is 5 times 3?")

    assert result.answer == "The answer is 15"
    assert call_count == 2  # Two calls: initial + after tool execution


@pytest.mark.asyncio
async def test_auto_execute_tools_false() -> None:
    """Test with auto_execute_tools=False - should return tool_calls without execution."""
    # Mock response - LLM requests tool
    response = ChatCompletion(
        id="test1",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content='{"answer": "Calling calculator"}',  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="Calculator",
                                arguments='{"operation": "add", "a": 10, "b": 20}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Should NOT execute tool, just return Prediction with tool_calls
    result = await predictor.aforward(auto_execute_tools=False, question="What is 10 plus 20?")

    # Should have tool_calls in the result
    assert result.native_tool_calls is not None
    assert len(result.native_tool_calls) == 1
    assert result.native_tool_calls[0]["name"] == "Calculator"
    assert result.native_tool_calls[0]["id"] == "call_123"
    assert '"operation": "add"' in result.native_tool_calls[0]["arguments"]

    # Should only make one call (no automatic execution)
    assert call_count == 1


def test_forward_with_auto_execute_tools_false() -> None:
    """Test sync forward() with auto_execute_tools=False."""
    # Mock response - LLM requests tool
    response = ChatCompletion(
        id="test1",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content='{"answer": "Calling calculator"}',  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_456",
                            type="function",
                            function=Function(
                                name="Calculator",
                                arguments='{"operation": "subtract", "a": 50, "b": 25}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Sync call with auto_execute_tools=False
    result = predictor.forward(auto_execute_tools=False, question="What is 50 minus 25?")

    # Should have tool_calls in the result
    assert result.native_tool_calls is not None
    assert len(result.native_tool_calls) == 1
    assert result.native_tool_calls[0]["name"] == "Calculator"

    # Should only make one call
    assert call_count == 1


def test_call_with_auto_execute_tools_false() -> None:
    """Test __call__() with auto_execute_tools=False."""
    # Mock response - LLM requests tool
    response = ChatCompletion(
        id="test1",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content='{"answer": "Calling calculator"}',  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_789",
                            type="function",
                            function=Function(
                                name="Calculator",
                                arguments='{"operation": "divide", "a": 100, "b": 5}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        return response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])

    # Call with auto_execute_tools=False
    result = predictor(auto_execute_tools=False, question="What is 100 divided by 5?")

    # Should have tool_calls in the result
    assert result.native_tool_calls is not None
    assert len(result.native_tool_calls) == 1
    assert result.native_tool_calls[0]["name"] == "Calculator"
