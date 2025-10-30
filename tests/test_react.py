"""Tests for ReAct module."""

import pytest
from conftest import make_mock_response
from pydantic import Field

from udspy import ConfirmationRequired, InputField, OutputField, ReAct, Signature, settings, tool


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


@tool(name="delete_file", description="Delete a file", require_confirmation=True)
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
    # Mock LLM responses for ReAct loop (using next_thought/next_tool_calls format)
    # First call: agent decides to call search tool
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI should search for information about Python\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "search", "args": {"query": "Python programming language"}}]'
    )

    # Second call: agent decides to finish
    react_finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI have the information I need\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    # Extract call: final answer extraction
    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nBased on the search results\n[[ ## answer ## ]]\nPython is a programming language"
    )

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return react_response
        elif call_count == 2:
            return react_finish_response
        else:
            return extract_response

    mock_aclient = settings.lm.client
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
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nFinish\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nCompleted\n[[ ## result ## ]]\nDone"
    )

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return react_response
        else:
            return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    # Create ReAct with string signature
    react = ReAct("query -> result", tools=[], enable_ask_to_user=False)

    result = await react.aforward(query="test")
    assert hasattr(result, "result")


@pytest.mark.asyncio
async def test_react_tool_confirmation() -> None:
    """Test ReAct with tool requiring confirmation."""
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nDelete the file\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "delete_file", "args": {"path": "/tmp/test.txt"}}]'
    )

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        return react_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    react = ReAct(QA, tools=[delete_file_tool], enable_ask_to_user=False)

    # Should raise ConfirmationRequired for confirmation
    with pytest.raises(ConfirmationRequired) as exc_info:
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
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nFinish\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nCompleted\n[[ ## answer ## ]]\nTest answer"
    )

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return react_response
        else:
            return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    react = ReAct(QA, tools=[], enable_ask_to_user=False)

    # Sync call
    result = react(question="Test?")
    assert result.answer == "Test answer"


def test_tool_with_require_confirmation_flag() -> None:
    """Test that tool require_confirmation flag is properly set."""
    assert delete_file_tool.require_confirmation is True
    assert search_tool.require_confirmation is False
    assert calculator_tool.require_confirmation is False


def test_tool_desc_and_args_aliases() -> None:
    """Test that Tool has desc and args aliases for DSPy compatibility."""
    assert search_tool.desc == search_tool.description
    assert "query" in search_tool.args
    assert search_tool.args["query"]["type"] == "string"  # JSON schema uses "string" not "str"


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
        require_confirmation=True,
    )

    # Mock response for initial call (LLM decides to delete)
    initial_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI will delete the file\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "delete_file", "args": {"path": "/tmp/test.txt"}}]'
    )

    # Mock response for after confirmation (LLM calls finish)
    finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nFile deleted successfully\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    # Mock extract response
    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nFile deleted\n[[ ## answer ## ]]\nFile was deleted successfully"
    )

    call_sequence = {"count": 0}

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        call_sequence["count"] += 1

        if call_sequence["count"] == 1:
            # Initial call - LLM decides to delete
            return initial_response
        elif call_sequence["count"] == 2:
            # After confirmation - LLM calls finish
            return finish_response
        else:
            # Extract call
            return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[tracked_tool], enable_ask_to_user=False)

    # Step 1: Initial execution should raise ConfirmationRequired
    try:
        await agent.aforward(question="Delete /tmp/test.txt")
        raise AssertionError("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        assert e.tool_call is not None
        assert e.tool_call.name == "delete_file"
        assert e.tool_call.args == {"path": "/tmp/test.txt"}
        assert call_count["count"] == 0  # Tool should NOT have been called yet
        saved_state = e

    # Step 2: User confirms - tool should be executed
    result = await agent.aresume("yes", saved_state)

    # Verify tool was actually called
    assert call_count["count"] == 1, "Tool should have been executed after confirmation"

    # Verify trajectory shows execution
    assert "observation_0" in result.trajectory
    assert "Deleted /tmp/test.txt" in result.trajectory["observation_0"]

    # Verify we have finish call in trajectory (iteration 1 uses normal execution, so check tool_calls_1)
    assert "tool_calls_1" in result.trajectory
    tool_calls_1 = result.trajectory["tool_calls_1"]
    assert len(tool_calls_1) > 0
    assert tool_calls_1[0]["name"] == "finish"


@pytest.mark.asyncio
async def test_user_feedback_triggers_re_reasoning() -> None:
    """Test that user feedback (not yes/no) causes LLM to re-reason."""
    # Mock response for initial ask_to_user
    ask_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI need more info\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "ask_to_user", "args": {"question": "What topic?"}}]'
    )

    # After user feedback, LLM uses search tool
    search_response = make_mock_response(
        "[[ ## next_thought ## ]]\nNow I'll search for Python\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "search", "args": {"query": "Python programming"}}]'
    )

    # Finally finish
    finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nDone\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nSearched for Python\n[[ ## answer ## ]]\nPython info"
    )

    call_sequence = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_sequence["count"] += 1

        if call_sequence["count"] == 1:
            return ask_response
        elif call_sequence["count"] == 2:
            return search_response
        elif call_sequence["count"] == 3:
            return finish_response
        else:
            return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[search_tool], enable_ask_to_user=True)

    # Step 1: Agent asks for clarification
    try:
        await agent.aforward(question="Tell me about it")
        raise AssertionError("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        assert e.tool_call is not None
        assert e.tool_call.name == "ask_to_user"
        saved_state = e

    # Step 2: User provides feedback (not yes/no)
    result = await agent.aresume("I want to know about Python", saved_state)

    # Verify LLM got the feedback and re-reasoned
    assert "observation_0" in result.trajectory
    assert "User feedback: I want to know about Python" in result.trajectory["observation_0"]

    # Verify LLM then used search tool (iteration 1 uses normal execution, so check tool_calls_1)
    assert "tool_calls_1" in result.trajectory
    tool_calls_1 = result.trajectory["tool_calls_1"]
    assert len(tool_calls_1) > 0
    assert tool_calls_1[0]["name"] == "search"
    assert tool_calls_1[0]["args"] == {"query": "Python programming"}


@pytest.mark.asyncio
async def test_react_with_string_signature() -> None:
    """Test ReAct with string signature format."""

    finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nReasoning\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nCompleted\n[[ ## result ## ]]\nTask completed"
    )

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return finish_response
        return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    # Test string signature: "input1, input2 -> output1, output2"
    agent = ReAct("task -> result", tools=[search_tool], enable_ask_to_user=False)

    result = await agent.aforward(task="Do something")

    assert isinstance(result.trajectory, dict)
    assert "result" in result


@pytest.mark.asyncio
async def test_react_resume_with_pending_tool_call() -> None:
    """Test ReAct resumption executes pending_tool_call (lines 265-287 in react.py)."""
    from udspy import respond_to_confirmation

    # First response: agent calls tool with require_confirmation
    response1 = make_mock_response(
        "[[ ## next_thought ## ]]\nI need to delete the file\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "delete_file", "args": {"path": "/tmp/test.txt"}}]'
    )

    # After resumption: agent finishes
    response2 = make_mock_response(
        "[[ ## next_thought ## ]]\nFile deleted\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nFile deleted\n[[ ## answer ## ]]\nFile was deleted"
    )

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        elif call_count["count"] == 2:
            return response2
        return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[delete_file_tool], max_iters=5)

    # First call - should raise ConfirmationRequired
    try:
        await agent.aforward(question="Delete /tmp/test.txt")
        pytest.fail("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        saved_state = e
        # Approve the confirmation
        respond_to_confirmation(e.confirmation_id, approved=True, status="approved")

    # Resume - this tests the pending_tool_call execution path (lines 265-287)
    result = await agent.aresume("yes", saved_state)

    assert isinstance(result, dict) or hasattr(result, "answer")
    # Verify the tool was executed
    assert "observation_0" in result.trajectory
    assert "Deleted" in result.trajectory["observation_0"]


@pytest.mark.asyncio
async def test_react_resume_pending_tool_call_with_exception() -> None:
    """Test pending_tool_call exception handling (lines 276-287 in react.py)."""
    from udspy import respond_to_confirmation

    @tool(name="failing_tool", description="Tool that fails", require_confirmation=True)
    def failing_tool(x: int = Field(...)) -> str:
        """Failing tool."""
        raise ValueError("Tool failed!")

    # Agent calls tool with require_confirmation
    response1 = make_mock_response(
        "[[ ## next_thought ## ]]\nCalling tool\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "failing_tool", "args": {"x": 1}}]'
    )

    # After tool failure, agent finishes
    response2 = make_mock_response(
        "[[ ## next_thought ## ]]\nTool failed\n"
        '[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]'
    )

    extract_response = make_mock_response(
        "[[ ## reasoning ## ]]\nTool failed\n[[ ## answer ## ]]\nOperation failed"
    )

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        elif call_count["count"] == 2:
            return response2
        return extract_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[failing_tool], max_iters=5)

    # First call - raises ConfirmationRequired
    try:
        await agent.aforward(question="Test")
        pytest.fail("Should have raised ConfirmationRequired")
    except ConfirmationRequired as e:
        saved_state = e
        respond_to_confirmation(e.confirmation_id, approved=True)

    # Resume - pending_tool_call execution hits exception path
    result = await agent.aresume("yes", saved_state)

    # Verify error was caught and logged as observation
    assert "observation_0" in result.trajectory
    assert "Traceback 'failing_tool'" in result.trajectory["observation_0"]
    assert "Tool failed" in result.trajectory["observation_0"]
