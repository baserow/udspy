"""Tests for History class."""

from unittest.mock import AsyncMock

import pytest

from udspy import History, InputField, OutputField, Predict, Signature, settings


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


def test_history_initialization() -> None:
    """Test History initialization."""
    # Empty history
    history = History()
    assert len(history) == 0
    assert history.messages == []

    # With initial messages
    messages = [{"role": "user", "content": "Hello"}]
    history = History(messages=messages)
    assert len(history) == 1
    assert history.messages == messages


def test_history_add_messages() -> None:
    """Test adding messages to history."""
    history = History()

    # Add user message
    history.add_user_message("Hello")
    assert len(history) == 1
    assert history.messages[0]["role"] == "user"
    assert history.messages[0]["content"] == "Hello"

    # Add assistant message
    history.add_assistant_message("Hi there!")
    assert len(history) == 2
    assert history.messages[1]["role"] == "assistant"
    assert history.messages[1]["content"] == "Hi there!"

    # Add system message
    history.add_system_message("You are helpful")
    assert len(history) == 3
    assert history.messages[2]["role"] == "system"
    assert history.messages[2]["content"] == "You are helpful"


def test_history_tool_result() -> None:
    """Test adding tool results to history."""
    history = History()

    history.add_tool_result(tool_call_id="call_123", content="Result: 42")

    assert len(history) == 1
    assert history.messages[0]["role"] == "tool"
    assert history.messages[0]["tool_call_id"] == "call_123"
    assert history.messages[0]["content"] == "Result: 42"


def test_history_clear() -> None:
    """Test clearing history."""
    history = History()
    history.add_user_message("Test")
    history.add_assistant_message("Response")

    assert len(history) == 2

    history.clear()
    assert len(history) == 0
    assert history.messages == []


def test_history_copy() -> None:
    """Test copying history."""
    history = History()
    history.add_user_message("Original")

    # Create copy
    copy = history.copy()

    # Modify copy
    copy.add_user_message("New message")

    # Original should be unchanged
    assert len(history) == 1
    assert len(copy) == 2


def test_history_repr() -> None:
    """Test History string representations."""
    history = History()
    history.add_user_message("Test message")

    # repr shows number of messages
    assert "History(1 messages)" in repr(history)

    # str shows formatted conversation
    str_repr = str(history)
    assert "History (1 messages)" in str_repr
    assert "[user]" in str_repr


@pytest.mark.asyncio
async def test_predict_with_history() -> None:
    """Test Predict with History for multi-turn conversation."""
    from conftest import make_mock_response

    responses = [
        "[[ ## answer ## ]]\nPython is a programming language",
        "[[ ## answer ## ]]\nKey features include simplicity and readability",
    ]
    call_count = 0

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        response = make_mock_response(responses[call_count])
        call_count += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA)
    history = History()

    # First turn
    result = await predictor.aforward(question="What is Python?", history=history)
    assert "Python" in result.answer

    # History should have system + user + assistant
    assert len(history) >= 2  # At least user and assistant

    # Second turn - history provides context
    result = await predictor.aforward(question="What are its key features?", history=history)
    assert "features" in result.answer.lower()

    # History should have more messages now
    assert len(history) >= 4  # Previous messages + new user + assistant


def test_predict_forward_with_history() -> None:
    """Test sync forward() with History."""
    from conftest import make_mock_response

    settings.lm.client.chat.completions.create = AsyncMock(
        return_value=make_mock_response("[[ ## answer ## ]]\nTest response")
    )

    predictor = Predict(QA)
    history = History()

    # Sync call with history
    result = predictor(question="Test question", history=history)

    assert "response" in result.answer.lower()
    # History should be updated
    assert len(history) >= 2


@pytest.mark.asyncio
async def test_predict_with_history_and_tools() -> None:
    """Test that history is updated correctly with tool calls and results."""
    from conftest import make_mock_response
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )
    from pydantic import Field

    from udspy import tool

    @tool(name="Calculator", description="Do math")
    def calculator(a: int = Field(...), b: int = Field(...)) -> int:
        """Add two numbers."""
        return a + b

    # Mock first response with tool call
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
                            id="call_123",
                            type="function",
                            function=Function(name="Calculator", arguments='{"a": 10, "b": 20}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response with final answer
    response2 = make_mock_response("[[ ## answer ## ]]\nThe result is 30")

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        return response2

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[calculator])
    history = History()

    result = await predictor.aforward(question="What is 10 + 20?", history=history)

    assert "30" in result.answer

    # Verify history contains the tool call and result
    tool_messages = [msg for msg in history.messages if msg["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_123"
    assert "30" in tool_messages[0]["content"]

    # Verify history contains assistant message with tool_calls
    assistant_with_tools = [
        msg for msg in history.messages if msg["role"] == "assistant" and "tool_calls" in msg
    ]
    assert len(assistant_with_tools) == 1


@pytest.mark.asyncio
async def test_predict_history_preserves_context() -> None:
    """Test that history correctly preserves conversation context across multiple calls."""
    from conftest import make_mock_response

    responses = [
        "[[ ## answer ## ]]\nPython is a programming language",
        "[[ ## answer ## ]]\nIt was created by Guido van Rossum",
    ]
    call_count = 0

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        response = make_mock_response(responses[call_count])
        call_count += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA)
    history = History()

    # First question
    result1 = await predictor.aforward(question="What is Python?", history=history)
    assert "Python" in result1.answer

    # Second question - should have context from first
    result2 = await predictor.aforward(question="Who created it?", history=history)
    assert "Guido" in result2.answer

    # Verify history has multiple turns
    assert len(history) >= 4  # At least 2 user + 2 assistant messages
