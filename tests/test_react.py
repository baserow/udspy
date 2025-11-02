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
    # Mock LLM responses for ReAct loop (using next_thought/next_tool_call format)
    # First call: agent decides to call search tool
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI should search for information about Python\n"
        '[[ ## next_tool_call ## ]]\n{"name": "search", "args": {"query": "Python programming language"}}'
    )

    # Second call: agent decides to finish
    react_finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nI have the information I need\n"
        '[[ ## next_tool_call ## ]]\n{"name": "finish", "args": {}}'
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
    react = ReAct(QA, tools=[search_tool])

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
        '[[ ## next_tool_call ## ]]\n{"name": "finish", "args": {}}'
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
    react = ReAct("query -> result", tools=[])

    result = await react.aforward(query="test")
    assert hasattr(result, "result")


@pytest.mark.asyncio
async def test_react_tool_confirmation() -> None:
    """Test ReAct with tool requiring confirmation."""
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nDelete the file\n"
        '[[ ## next_tool_call ## ]]\n{"name": "delete_file", "args": {"path": "/tmp/test.txt"}}'
    )

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        return react_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    react = ReAct(QA, tools=[delete_file_tool])

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
    # No longer storing iteration separately - use len(trajectory)


def test_react_forward_sync() -> None:
    """Test sync forward() method."""
    react_response = make_mock_response(
        "[[ ## next_thought ## ]]\nFinish\n"
        '[[ ## next_tool_call ## ]]\n{"name": "finish", "args": {}}'
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

    react = ReAct(QA, tools=[])

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
async def test_react_with_string_signature() -> None:
    """Test ReAct with string signature format."""

    finish_response = make_mock_response(
        "[[ ## next_thought ## ]]\nReasoning\n"
        '[[ ## next_tool_call ## ]]\n{"name": "finish", "args": {}}'
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
    agent = ReAct("task -> result", tools=[search_tool])

    result = await agent.aforward(task="Do something")

    # Trajectory is now a list of episodes
    assert isinstance(result.trajectory, list)
    assert "result" in result
