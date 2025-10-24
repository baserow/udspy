"""Tests for History class."""

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

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
    # Mock streaming responses
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
                        content="[[ ## answer ## ]]\nPython is a programming language"
                    ),
                    finish_reason=None,
                )
            ],
        )
    ]

    second_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## answer ## ]]\nKey features include simplicity and readability"
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
    # Mock streaming response
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\nTest response"),
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

    predictor = Predict(QA)
    history = History()

    # Sync call with history
    result = predictor(question="Test question", history=history)

    assert "response" in result.answer.lower()
    # History should be updated
    assert len(history) >= 2


def test_history_invalid_type_error() -> None:
    """Test that passing non-History object raises error."""

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            yield ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[Choice(index=0, delta=ChoiceDelta(content="test"), finish_reason=None)],
            )

        return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA)

    # Passing non-History object should raise TypeError
    with pytest.raises(TypeError, match="history must be a History object"):
        predictor(question="Test", history={"invalid": "type"})
