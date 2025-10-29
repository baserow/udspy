"""Tests for @tool decorator and automatic tool execution."""

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


def test_tool_decorator() -> None:
    """Test @tool decorator creates Tool object."""
    assert calculator.name == "Calculator"
    assert calculator.description == "Perform arithmetic operations"
    assert "operation" in calculator.args_schema["properties"]
    assert "a" in calculator.args_schema["properties"]
    assert "b" in calculator.args_schema["properties"]


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

    # Mock second response - LLM provides final answer after seeing tool result
    second_response = make_mock_response("[[ ## answer ## ]]\nThe answer is 15")

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call - return tool call
            return first_response
        else:
            # Second call - return final answer
            return second_response

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


def test_predict_stores_tool_schemas() -> None:
    """Test Predict stores tool schemas correctly."""
    predictor = Predict(QA, tools=[calculator])

    # Verify tool is stored in tools dict
    assert "Calculator" in predictor.tools
    assert predictor.tools["Calculator"] == calculator

    # Verify the tool schema is in tool_schemas
    assert len(predictor.tool_schemas) == 1
    tool_schema = predictor.tool_schemas[0]
    assert tool_schema["type"] == "function"
    assert tool_schema["function"]["name"] == "Calculator"


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

    @tool(name="FailingTool", description="A tool that fails")
    def failing_tool(value: int = Field(description="A value")) -> int:
        raise ValueError("Tool failed!")

    # Mock first response - LLM requests failing tool
    first_response = ChatCompletion(
        id="test",
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
                            id="call_456",
                            type="function",
                            function=Function(
                                name="FailingTool",
                                arguments='{"value": 42}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response - LLM provides answer after seeing error
    second_response = make_mock_response("[[ ## answer ## ]]\nThe tool encountered an error")

    call_count = 0
    messages_log = []

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        messages_log.append(kwargs.get("messages", []))

        if call_count == 1:
            return first_response
        else:
            return second_response

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
