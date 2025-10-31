"""Tests for edge cases and error handling."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from conftest import make_mock_response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import Field

from udspy import InputField, OutputField, Predict, Prediction, Signature, settings, tool
from udspy.history import History
from udspy.streaming import OutputStreamChunk


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


@tool(name="Calculator", description="Simple calculator")
def calculator(a: float = Field(...), b: float = Field(...)) -> float:
    """Add two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_max_turns_reached_error() -> None:
    """Test that RuntimeError is raised when max turns is reached with pending tool calls."""
    response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\nPlaceholder",  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="Calculator", arguments='{"a": 1, "b": 2}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(return_value=response)

    predictor = Predict(QA, tools=[calculator], max_turns=2)

    with pytest.raises(RuntimeError, match="Max turns .* reached without final answer"):
        await predictor.aforward(question="What is 1+2?")


@pytest.mark.asyncio
async def test_tool_not_found_error() -> None:
    """Test error handling when LLM calls a non-existent tool."""
    response1 = ChatCompletion(
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
                            id="call_1",
                            type="function",
                            function=Function(name="NonExistentTool", arguments='{"x": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response2 = make_mock_response("[[ ## answer ## ]]\nI encountered an error.")

    mock_aclient = settings.lm.client
    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return response1
        return response2

    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])
    result = await predictor.aforward(question="Test")

    assert isinstance(result, Prediction)
    assert "error" in result.answer.lower() or result.answer


@pytest.mark.asyncio
async def test_tool_execution_exception() -> None:
    """Test error handling when tool execution raises an exception."""

    @tool(name="FailingTool", description="A tool that always fails")
    def failing_tool(x: int = Field(...)) -> int:
        """Failing tool."""
        raise ValueError("Tool execution failed")

    response1 = ChatCompletion(
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
                            id="call_1",
                            type="function",
                            function=Function(name="FailingTool", arguments='{"x": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response2 = make_mock_response("[[ ## answer ## ]]\nThe tool failed.")

    mock_aclient = settings.lm.client
    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return response1
        return response2

    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[failing_tool])
    result = await predictor.aforward(question="Test")

    assert isinstance(result, Prediction)
    assert result.answer


@pytest.mark.asyncio
async def test_history_update_with_tools() -> None:
    """Test that history is properly updated with tool calls and results."""
    history = History()

    response1 = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\nCalculating",  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="Calculator", arguments='{"a": 5, "b": 3}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response2 = make_mock_response("[[ ## answer ## ]]\n8")

    mock_aclient = settings.lm.client
    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return response1
        return response2

    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])
    result = await predictor.aforward(history=history, question="What is 5+3?")

    assert isinstance(result, Prediction)
    assert result.answer == "8"

    # Verify history was updated
    assert len(history.messages) >= 3
    # Should have: user message, assistant with tool_calls, tool result, assistant answer
    tool_call_msg = None
    tool_result_msg = None

    for msg in history.messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            tool_call_msg = msg
        if msg["role"] == "tool":
            tool_result_msg = msg

    assert tool_call_msg is not None
    assert tool_result_msg is not None
    assert "8" in tool_result_msg["content"]


@pytest.mark.asyncio
async def test_stream_chunk_repr() -> None:
    """Test OutputStreamChunk __repr__ method for complete and streaming states."""
    predictor = Predict(QA)

    streaming_chunk = OutputStreamChunk(
        predictor,
        field_name="answer",
        delta="Paris",
        content="Paris",
        is_complete=False,
    )
    assert "streaming" in repr(streaming_chunk)
    assert "answer" in repr(streaming_chunk)
    assert "Paris" in repr(streaming_chunk)

    complete_chunk = OutputStreamChunk(
        predictor, field_name="answer", delta="", content="Paris", is_complete=True
    )
    assert "complete" in repr(complete_chunk)


@pytest.mark.asyncio
async def test_prediction_attribute_error() -> None:
    """Test that accessing non-existent Prediction attributes raises AttributeError."""
    pred = Prediction(answer="Paris")

    assert pred.answer == "Paris"
    assert pred["answer"] == "Paris"

    # Test __setattr__
    pred.new_field = "New Value"
    assert pred.new_field == "New Value"
    assert pred["new_field"] == "New Value"

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = pred.nonexistent


@pytest.mark.asyncio
async def test_astream_exception_propagation() -> None:
    """Test that exceptions during streaming are properly propagated."""

    async def failing_execute(*, stream: bool = False, **inputs):  # type: ignore[no-untyped-def]
        raise ValueError("Execution failed")

    predictor = Predict(QA)
    predictor.aexecute = failing_execute  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Execution failed"):
        async for _ in predictor.astream(question="Test"):
            pass


def test_sync_forward_from_async_context() -> None:
    """Test that calling sync forward() from async context raises RuntimeError."""

    async def test_call() -> None:
        predictor = Predict(QA)
        with pytest.raises(RuntimeError, match="Cannot call.*from async context"):
            predictor.forward(question="Test")

    asyncio.run(test_call())


@pytest.mark.asyncio
async def test_no_tools_available_but_tool_calls_present() -> None:
    """Test handling when LLM returns tool calls but no tools are available."""
    response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\nNo tool available",  # Add content to avoid AdapterParseError
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="SomeTool", arguments='{"x": 1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(return_value=response)

    # Create predictor with no tools but auto_execute_tools=True
    predictor = Predict(QA)

    # Should break out of loop since tool_callables is empty
    result = await predictor.aforward(auto_execute_tools=True, question="Test")

    assert isinstance(result, Prediction)
    assert "native_tool_calls" in result


@pytest.mark.asyncio
async def test_invalid_json_in_tool_arguments() -> None:
    """Test handling of malformed JSON in tool call arguments."""
    from udspy.module.react import ReAct

    @tool(name="TestTool", description="Test tool")
    def test_tool(x: int = Field(...)) -> str:
        """Test tool."""
        return f"Result: {x}"

    class SimpleTask(Signature):
        """Perform a simple task."""

        task: str = InputField()
        result: str = OutputField()

    # Mock response with malformed JSON in tool arguments - note the invalid JSON in args
    response = make_mock_response(
        "[[ ## next_thought ## ]]\nI'll use the test tool\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "TestTool", "args": {"x": invalid_json}}]'  # Malformed JSON in content
    )

    response2 = make_mock_response(
        "[[ ## next_thought ## ]]\nLet me try again\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    response3 = make_mock_response(
        "[[ ## next_thought ## ]]\nFinished\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    response4 = make_mock_response(
        "[[ ## reasoning ## ]]\nCompleted\n[[ ## result ## ]]\nCompleted"
    )

    mock_aclient = settings.lm.client
    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return response
        if call_count == 2:
            return response2
        if call_count == 3:
            return response3
        return response4

    mock_aclient.chat.completions.create = mock_create

    react = ReAct(SimpleTask, tools=[test_tool], max_iters=5)
    result = await react.aforward(task="Test task")

    # Should handle malformed JSON gracefully and continue
    # The JSON decode error is logged and next_tool_calls becomes empty list
    assert isinstance(result, Prediction)
    assert "trajectory" in result
    # Verify the malformed JSON was handled (first episode should have empty tool_calls)
    assert isinstance(result.trajectory, list)
    assert len(result.trajectory) >= 1
    assert result.trajectory[0]["tool_calls"] == []
    # Agent should have continued to next iteration (second episode)
    assert len(result.trajectory) >= 2
    assert result.trajectory[1]["thought"] == "Let me try again"
