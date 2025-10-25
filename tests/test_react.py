"""Tests for ReAct module."""

import json

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta, ChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as ToolCallFunction,
)
from pydantic import Field

from udspy import HumanInTheLoopRequired, InputField, OutputField, ReAct, Signature, settings, tool


# Test tools
@tool(name="search", description="Search for information")
def search_tool(query: str = Field(description="Search query")) -> str:
    """Mock search tool."""
    return f"Search results for: {query}"


@tool(name="calculator", description="Perform calculations")
def calculator_tool(expression: str = Field(description="Math expression")) -> str:
    """Mock calculator tool."""
    if expression == "2+2":
        return "4"
    return "42"


@tool(name="delete_file", description="Delete a file", interruptible=True)
def delete_file_tool(path: str = Field(description="File path")) -> str:
    """Mock destructive tool requiring confirmation."""
    return f"Deleted {path}"


class QA(Signature):
    """Answer questions using available tools."""

    question: str = InputField()
    answer: str = OutputField()


@pytest.mark.asyncio
async def test_react_basic_execution() -> None:
    """Test basic ReAct execution with a simple tool."""
    # Mock LLM responses for ReAct loop with native tool calling
    # First call: agent decides to call search tool
    react_chunks = [
        ChatCompletionChunk(
            id="test1",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nI should search for information about Python",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_search123",
                                function=ToolCallFunction(
                                    name="search",
                                    arguments=json.dumps({"query": "Python programming language"}),
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # Second call: agent decides to finish
    react_chunks_finish = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nI have the information I need",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_finish456",
                                function=ToolCallFunction(name="finish", arguments="{}"),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # Extract call: final answer extraction
    extract_chunks = [
        ChatCompletionChunk(
            id="test3",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nBased on the search results\n[[ ## answer ## ]]\nPython is a programming language"
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

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in react_chunks:
                    yield chunk

            return stream()
        elif call_count == 2:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in react_chunks_finish:
                    yield chunk

            return stream()
        else:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in extract_chunks:
                    yield chunk

            return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    # Create ReAct with search tool
    react = ReAct(QA, tools=[search_tool], enable_ask_to_user=False)

    # Execute
    result = await react.aforward(question="What is Python?")

    # Verify result
    assert "Python" in result.answer
    assert "trajectory" in result
    assert call_count >= 2  # At least react and extract calls


@pytest.mark.asyncio
async def test_react_string_signature() -> None:
    """Test ReAct with string signature format."""
    react_chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nFinish",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_finish789",
                                function=ToolCallFunction(name="finish", arguments="{}"),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    extract_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## result ## ]]\nDone"),
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

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in react_chunks:
                    yield chunk

            return stream()
        else:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in extract_chunks:
                    yield chunk

            return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    # Create ReAct with string signature
    react = ReAct("query -> result", tools=[], enable_ask_to_user=False)

    result = await react.aforward(query="test")
    assert hasattr(result, "result")


@pytest.mark.asyncio
async def test_react_tool_confirmation() -> None:
    """Test ReAct with tool requiring confirmation."""
    react_chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nDelete the file",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_delete123",
                                function=ToolCallFunction(
                                    name="delete_file",
                                    arguments=json.dumps({"path": "/tmp/test.txt"}),
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            for chunk in react_chunks:
                yield chunk

        return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    react = ReAct(QA, tools=[delete_file_tool], enable_ask_to_user=False)

    # Should raise HumanInTheLoopRequired for confirmation
    with pytest.raises(HumanInTheLoopRequired) as exc_info:
        await react.aforward(question="Delete the test file")

    assert "Confirm execution" in exc_info.value.question
    assert "delete_file" in exc_info.value.question
    # Verify exception has rich context
    assert exc_info.value.tool_call is not None
    assert exc_info.value.tool_call.name == "delete_file"
    assert exc_info.value.tool_call.args == {"path": "/tmp/test.txt"}
    assert exc_info.value.context.get("trajectory") is not None
    assert exc_info.value.context.get("iteration") is not None


def test_react_forward_sync() -> None:
    """Test sync forward() method."""
    react_chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nFinish",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_sync123",
                                function=ToolCallFunction(name="finish", arguments="{}"),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    extract_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\nTest answer"),
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

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in react_chunks:
                    yield chunk

            return stream()
        else:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in extract_chunks:
                    yield chunk

            return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    react = ReAct(QA, tools=[], enable_ask_to_user=False)

    # Sync call
    result = react(question="Test?")
    assert result.answer == "Test answer"


def test_tool_with_interruptible_flag() -> None:
    """Test that tool interruptible flag is properly set."""
    assert delete_file_tool.interruptible is True
    assert search_tool.interruptible is False
    assert calculator_tool.interruptible is False


def test_tool_desc_and_args_aliases() -> None:
    """Test that Tool has desc and args aliases for DSPy compatibility."""
    assert search_tool.desc == search_tool.description
    assert "query" in search_tool.args
    assert "str" in search_tool.args["query"]


@pytest.mark.asyncio
async def test_tool_execution_after_confirmation() -> None:
    """Test that confirmed tools are actually executed and results added to trajectory."""
    # Mock tool that tracks if it was called
    call_count = {"count": 0}

    def tracked_delete(path: str = Field(description="File path")) -> str:
        call_count["count"] += 1
        return f"Deleted {path}"

    from udspy.tool import Tool

    tracked_tool = Tool(
        func=tracked_delete,
        name="delete_file",
        description="Delete a file",
        interruptible=True,
    )

    # Mock chunks for initial call (LLM decides to delete)
    initial_chunks = [
        ChatCompletionChunk(
            id="test1",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nI will delete the file",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_delete1",
                                function=ToolCallFunction(
                                    name="delete_file",
                                    arguments=json.dumps({"path": "/tmp/test.txt"}),
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # Mock chunks for after confirmation (LLM calls finish)
    finish_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nFile deleted successfully",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_finish1",
                                function=ToolCallFunction(name="finish", arguments="{}"),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # Mock extract chunks
    extract_chunks = [
        ChatCompletionChunk(
            id="test3",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\nFile was deleted successfully"),
                    finish_reason=None,
                )
            ],
        )
    ]

    call_sequence = {"count": 0}

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        call_sequence["count"] += 1

        if call_sequence["count"] == 1:
            # Initial call - LLM decides to delete

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in initial_chunks:
                    yield chunk

            return stream()
        elif call_sequence["count"] == 2:
            # After confirmation - LLM calls finish

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in finish_chunks:
                    yield chunk

            return stream()
        else:
            # Extract call

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in extract_chunks:
                    yield chunk

            return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[tracked_tool], enable_ask_to_user=False)

    # Step 1: Initial execution should raise HumanInTheLoopRequired
    try:
        await agent.aforward(question="Delete /tmp/test.txt")
        raise AssertionError("Should have raised HumanInTheLoopRequired")
    except HumanInTheLoopRequired as e:
        assert e.tool_call is not None
        assert e.tool_call.name == "delete_file"
        assert e.tool_call.args == {"path": "/tmp/test.txt"}
        assert call_count["count"] == 0  # Tool should NOT have been called yet
        saved_state = e

    # Step 2: User confirms - tool should be executed
    result = await agent.aresume_after_user_input("yes", saved_state)

    # Verify tool was actually called
    assert call_count["count"] == 1, "Tool should have been executed after confirmation"

    # Verify trajectory shows execution
    assert "observation_0" in result.trajectory
    assert "Deleted /tmp/test.txt" in result.trajectory["observation_0"]

    # Verify we have finish call in trajectory
    assert "tool_name_1" in result.trajectory
    assert result.trajectory["tool_name_1"] == "finish"


@pytest.mark.asyncio
async def test_user_feedback_triggers_re_reasoning() -> None:
    """Test that user feedback (not yes/no) causes LLM to re-reason."""
    # Mock chunks for initial ask_to_user
    ask_chunks = [
        ChatCompletionChunk(
            id="test1",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nI need more info",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_ask1",
                                function=ToolCallFunction(
                                    name="ask_to_user",
                                    arguments=json.dumps({"question": "What topic?"}),
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # After user feedback, LLM uses search tool
    search_chunks = [
        ChatCompletionChunk(
            id="test2",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nNow I'll search for Python",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_search1",
                                function=ToolCallFunction(
                                    name="search",
                                    arguments=json.dumps({"query": "Python programming"}),
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    # Finally finish
    finish_chunks = [
        ChatCompletionChunk(
            id="test3",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content="[[ ## reasoning ## ]]\nDone",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_finish1",
                                function=ToolCallFunction(name="finish", arguments="{}"),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    extract_chunks = [
        ChatCompletionChunk(
            id="test4",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="[[ ## answer ## ]]\nPython info"),
                    finish_reason=None,
                )
            ],
        )
    ]

    call_sequence = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_sequence["count"] += 1

        if call_sequence["count"] == 1:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in ask_chunks:
                    yield chunk

            return stream()
        elif call_sequence["count"] == 2:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in search_chunks:
                    yield chunk

            return stream()
        elif call_sequence["count"] == 3:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in finish_chunks:
                    yield chunk

            return stream()
        else:

            async def stream():  # type: ignore[no-untyped-def]
                for chunk in extract_chunks:
                    yield chunk

            return stream()

    mock_aclient = settings.aclient
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[search_tool], enable_ask_to_user=True)

    # Step 1: Agent asks for clarification
    try:
        await agent.aforward(question="Tell me about it")
        raise AssertionError("Should have raised HumanInTheLoopRequired")
    except HumanInTheLoopRequired as e:
        assert e.tool_call is not None
        assert e.tool_call.name == "ask_to_user"
        saved_state = e

    # Step 2: User provides feedback (not yes/no)
    result = await agent.aresume_after_user_input("I want to know about Python", saved_state)

    # Verify LLM got the feedback and re-reasoned
    assert "observation_0" in result.trajectory
    assert "User feedback: I want to know about Python" in result.trajectory["observation_0"]

    # Verify LLM then used search tool (iteration 1)
    assert "tool_name_1" in result.trajectory
    assert result.trajectory["tool_name_1"] == "search"
    assert result.trajectory["tool_args_1"] == {"query": "Python programming"}
