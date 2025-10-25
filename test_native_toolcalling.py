"""Test native tool calling with ReAct."""

import asyncio
import logging

from openai import AsyncOpenAI
from pydantic import Field

from udspy import HumanInTheLoopRequired, InputField, OutputField, ReAct, Signature, settings, tool

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')


# Configure with local LLM
local_client = AsyncOpenAI(
    api_key="sk-no-key-required",
    base_url="http://localhost:11434/v1",
)
settings.configure(
    aclient=local_client,
    model="gpt-oss:120b-cloud",
)


# Define test tools
@tool(name="calculator", description="Calculate mathematical expressions")
def calculator(expression: str = Field(description="Math expression to evaluate")) -> str:
    """Simple calculator tool."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(name="search", description="Search for information")
def search(query: str = Field(description="Search query")) -> str:
    """Mock search tool."""
    responses = {
        "python": "Python is a high-level programming language",
        "react": "React is a JavaScript library for UIs",
    }
    for key, value in responses.items():
        if key in query.lower():
            return value
    return f"No results for: {query}"


@tool(
    name="delete_file",
    description="Delete a file (requires confirmation)",
    interruptible=True,
)
def delete_file(path: str = Field(description="File path")) -> str:
    """Delete a file - requires confirmation."""
    return f"Deleted {path}"


# Define task signature
class QA(Signature):
    """Answer questions using available tools."""

    question: str = InputField()
    answer: str = OutputField()


async def test_basic():
    """Test basic ReAct with native tool calling."""
    print("=== Test 1: Basic calculation ===\n")

    agent = ReAct(QA, tools=[calculator], enable_ask_to_user=False, max_iters=3)

    try:
        result = await agent.aforward(question="What is 2 + 2?")
        print(f"Answer: {result.answer}")
        print("\nTrajectory:")
        for key, value in result.trajectory.items():
            print(f"  {key}: {value}")
        print("\n✓ Test 1 passed\n")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")
        import traceback
        traceback.print_exc()


async def test_tool_confirmation():
    """Test tool that requires confirmation."""
    print("=== Test 2: Tool confirmation ===\n")

    agent = ReAct(QA, tools=[delete_file], enable_ask_to_user=False, max_iters=3)

    try:
        await agent.aforward(question="Delete /tmp/test.txt")
        print("✗ Test 2 failed: Should have raised HumanInTheLoopRequired\n")
    except HumanInTheLoopRequired as e:
        print(f"Confirmation request: {e.question}")
        print(f"Tool name: {e.tool_name}")
        print(f"Tool args: {e.tool_args}")
        print(f"Has trajectory: {bool(e.trajectory)}")
        print(f"Has iteration: {e.iteration is not None}")
        print("\n✓ Test 2 passed\n")
    except Exception as e:
        print(f"✗ Test 2 failed with unexpected error: {e}\n")
        import traceback
        traceback.print_exc()


async def test_ask_to_user():
    """Test ask_to_user tool."""
    print("=== Test 3: ask_to_user ===\n")

    agent = ReAct(QA, tools=[search], enable_ask_to_user=True, max_iters=3)

    try:
        # This should trigger ask_to_user due to ambiguous question
        result = await agent.aforward(question="Tell me about it")
        print("✗ Test 3 failed: Should have raised HumanInTheLoopRequired\n")
    except HumanInTheLoopRequired as e:
        print(f"Question from agent: {e.question}")
        print(f"Tool name: {e.tool_name}")
        print(f"Has trajectory: {bool(e.trajectory)}")

        # Now test resume
        print("\nResuming with user response...")
        try:
            result = await agent.aresume_after_user_input("Python programming", e)
            print(f"Answer after resume: {result.answer}")
            print("\n✓ Test 3 passed\n")
        except Exception as resume_error:
            print(f"✗ Test 3 failed during resume: {resume_error}\n")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"✗ Test 3 failed with unexpected error: {e}\n")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    await test_basic()
    await test_tool_confirmation()
    await test_ask_to_user()
    print("=== All tests complete ===")


if __name__ == "__main__":
    asyncio.run(main())
