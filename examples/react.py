"""ReAct (Reasoning and Acting) example.

This example demonstrates the ReAct module for building agents that can:
- Use multiple tools to accomplish tasks
- Reason through multi-step problems
- Ask users for clarification when needed
- Request confirmation for destructive operations
"""

import asyncio

from pydantic import Field

from udspy import HumanInTheLoopRequired, InputField, OutputField, ReAct, Signature, settings, tool

settings.configure(
    model="gpt-oss:120b-cloud",
    base_url="http://localhost:11434/v1",
)


# Define tools for the agent
@tool(name="search", description="Search for information on the internet")
def search(query: str = Field(description="The search query")) -> str:
    """Mock search tool - in a real application, this would call a search API."""
    # Simulate search results
    search_results = {
        "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "react": "React is a JavaScript library for building user interfaces, developed by Facebook.",
        "weather tokyo": "Tokyo weather: Sunny, 22°C (72°F), humidity 60%",
    }

    query_lower = query.lower()
    for key, value in search_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {value}"

    return f"No specific results found for '{query}'. Try a different search term."


@tool(name="calculator", description="Perform mathematical calculations")
def calculator(expression: str = Field(description="Mathematical expression to evaluate")) -> str:
    """Simple calculator tool."""
    try:
        # Safe evaluation of simple expressions
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool(name="delete_file", description="Delete a file (requires confirmation)", interruptible=True)
def delete_file(path: str = Field(description="Path to the file to delete")) -> str:
    """Mock file deletion tool - requires user confirmation."""
    # In a real application, this would actually delete the file
    return f"File '{path}' has been deleted successfully."


# Define the task signature
class ResearchTask(Signature):
    """Research a topic and provide a comprehensive answer using available tools."""

    question: str = InputField(description="The research question or task")
    answer: str = OutputField(description="Comprehensive answer based on tool usage")


async def basic_research_example():
    """Basic example: ReAct agent researches a topic using search."""
    print("=== Basic Research Example ===\n")

    # Create ReAct agent with search and calculator tools
    agent = ReAct(
        ResearchTask,
        tools=[search, calculator],
        enable_ask_to_user=False,  # Disable for this simple example
        max_iters=5,
    )

    # Ask a question that requires tool use
    question = "What is Python and how many letters are in the word?"
    print(f"Question: {question}\n")

    result = await agent.aforward(question=question)

    print(f"Answer: {result.answer}\n")
    print(
        f"Number of reasoning steps: {len([k for k in result.trajectory if k.startswith('reasoning_')])}\n"
    )


async def user_clarification_example():
    """Example showing how the agent asks for user clarification."""
    print("=== User Clarification Example ===\n")

    # Create agent with ask_to_user enabled
    agent = ReAct(
        ResearchTask,
        tools=[search, calculator],
        enable_ask_to_user=True,
        max_iters=5,
    )

    # Ambiguous question
    question = "Tell me about it"
    print(f"Question: {question}\n")

    try:
        result = await agent.aforward(question=question)
        print(f"Answer: {result.answer}\n")
    except HumanInTheLoopRequired as e:
        # Agent needs clarification
        print(f"Agent asks: {e.question}\n")

        # Simulate user providing clarification
        user_response = "I want to know about the Python programming language"
        print(f"User responds: {user_response}\n")

        # Resume execution with user's response
        try:
            result = await agent.aresume(user_response, e)
            if hasattr(result, "answer"):
                print(f"Answer: {result.answer}\n")
            else:
                print("(Resume completed but LLM didn't format response correctly)\n")
        except Exception as resume_error:
            print(
                f"(Note: Resume encountered LLM formatting issue: {type(resume_error).__name__})\n"
            )


async def tool_confirmation_example():
    """Example showing tool confirmation for destructive operations."""
    print("=== Tool Confirmation Example ===\n")

    # Create agent with a destructive tool
    agent = ReAct(
        ResearchTask,
        tools=[search, delete_file],
        enable_ask_to_user=False,
        max_iters=5,
    )

    question = "Delete the file /tmp/old_data.txt"
    print(f"Question: {question}\n")

    try:
        result = await agent.aforward(question=question)
        print(f"Answer: {result.answer}\n")
    except HumanInTheLoopRequired as e:
        # Agent needs confirmation for destructive operation
        print(f"Confirmation needed: {e.question}\n")

        # Simulate user confirming
        user_response = "yes"
        print(f"User confirms: {user_response}\n")

        # Resume execution
        try:
            result = await agent.aresume(user_response, e)
            if hasattr(result, "answer"):
                print(f"Answer: {result.answer}\n")
            else:
                print("(Confirmation accepted but LLM didn't format response correctly)\n")
        except Exception as resume_error:
            print(
                f"(Note: Resume encountered LLM formatting issue: {type(resume_error).__name__})\n"
            )


async def string_signature_example():
    """Example using string signature format for quick prototyping."""
    print("=== String Signature Example ===\n")

    # Create agent with simple string signature
    agent = ReAct(
        "task -> result",  # Simple signature format
        tools=[search, calculator],
        enable_ask_to_user=False,
        max_iters=5,
    )

    task = "Search for React and calculate 2 + 2"
    print(f"Task: {task}\n")

    result = await agent.aforward(task=task)
    print(f"Result: {result.result}\n")


async def main():
    """Run all examples."""
    # Basic research example
    await basic_research_example()

    # User clarification example
    await user_clarification_example()

    # Tool confirmation example
    await tool_confirmation_example()

    # String signature example
    await string_signature_example()

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
