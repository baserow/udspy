"""Example demonstrating dynamic tool management with module callbacks.

This example shows how tools can dynamically add other tools during execution
using the @module_callback decorator. The agent starts with basic tools and
can load specialized tools on demand.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

from pydantic import Field

import udspy
from udspy import (
    InputField,
    OutputField,
    ReAct,
    Signature,
    module_callback,
    tool,
)


# Define specialized tools that will be loaded dynamically
@tool(name="get_weather", description="Get weather for a location")
def get_weather(location: str = Field(..., description="City name")) -> str:
    """Get current weather (mock implementation)."""
    # In a real app, this would call a weather API
    return f"Weather in {location}: Sunny, 72°F"


@tool(name="get_time", description="Get current time for a location")
def get_time(location: str = Field(..., description="City name")) -> str:
    """Get current time (mock implementation)."""
    # In a real app, this would handle timezones
    return f"Current time in {location}: 2:30 PM"


@tool(name="search_web", description="Search the web for information")
def search_web(query: str = Field(..., description="Search query")) -> str:
    """Search web (mock implementation)."""
    # In a real app, this would call a search API
    return f"Search results for '{query}': Found relevant information..."


# Meta-tool that loads specialized tools based on category
@tool(name="load_tools", description="Load specialized tools for a category")
def load_tools(
    category: str = Field(..., description="Tool category: weather, time, or search"),
) -> callable:  # type: ignore[valid-syntax]
    """Load specialized tools dynamically.

    This tool returns a module callback that adds new tools to the agent.
    """

    @module_callback
    def add_tools_callback(context):
        # Get current tools (excluding built-ins)
        current_tools = [
            t
            for t in context.module.tools.values()
            if t.name not in ("finish", "ask_to_user", "load_tools")
        ]

        # Select new tools based on category
        new_tools = []
        if category == "weather":
            new_tools = [get_weather]
        elif category == "time":
            new_tools = [get_time]
        elif category == "search":
            new_tools = [search_web]
        elif category == "all":
            new_tools = [get_weather, get_time, search_web]
        else:
            return f"Unknown category '{category}'. Available: weather, time, search, all"

        # Combine with existing tools
        all_tools = current_tools + new_tools

        # Reinitialize module with all tools
        context.module.init_module(tools=all_tools)

        tool_names = [t.name if hasattr(t, "name") else t.__name__ for t in new_tools]
        return f"Loaded {len(new_tools)} tools for category '{category}': {', '.join(tool_names)}"

    return add_tools_callback


class Question(Signature):
    """Answer questions using available tools. Load specialized tools if needed."""

    question: str = InputField()
    answer: str = OutputField()


def basic_example() -> None:
    """Basic example showing dynamic tool loading."""
    print("=== Basic Dynamic Tool Loading Example ===\n")

    # Create agent with only the load_tools meta-tool
    agent = ReAct(Question, tools=[load_tools])

    # Agent should load weather tools and use them
    question = "What's the weather in Tokyo?"
    print(f"Question: {question}\n")

    result = agent(question=question)
    print(f"\nAnswer: {result.answer}\n")


def multi_category_example() -> None:
    """Example showing loading multiple tool categories."""
    print("\n=== Multi-Category Tool Loading Example ===\n")

    agent = ReAct(Question, tools=[load_tools])

    # Question requiring multiple tool categories
    question = "What's the weather in London and what time is it there?"
    print(f"Question: {question}\n")

    result = agent(question=question)
    print(f"\nAnswer: {result.answer}\n")


def load_all_example() -> None:
    """Example showing loading all available tools at once."""
    print("\n=== Load All Tools Example ===\n")

    agent = ReAct(Question, tools=[load_tools])

    # Complex question requiring multiple tools
    question = "Load all available tools, then tell me the weather in Paris and search for information about the Eiffel Tower"
    print(f"Question: {question}\n")

    result = agent(question=question)
    print(f"\nAnswer: {result.answer}\n")


def inspect_tools_example() -> None:
    """Example showing how to inspect available tools."""
    print("\n=== Inspect Tools Example ===\n")

    agent = ReAct(Question, tools=[load_tools])

    print("Initial tools:")
    for tool_name in agent.tools:
        print(f"  - {tool_name}")

    # Load weather tools
    print("\nLoading weather tools...")
    agent(question="Load weather tools and tell me the weather in Seattle")

    print("\nTools after loading weather:")
    for tool_name in agent.tools:
        print(f"  - {tool_name}")


def error_handling_example() -> None:
    """Example showing error handling for invalid categories."""
    print("\n=== Error Handling Example ===\n")

    agent = ReAct(Question, tools=[load_tools])

    question = "Load tools for category 'invalid_category'"
    print(f"Question: {question}\n")

    result = agent(question=question)
    print(f"\nAnswer: {result.answer}\n")


if __name__ == "__main__":
    # Configure from environment variables
    udspy.settings.configure()

    # Run examples
    basic_example()
    multi_category_example()
    load_all_example()
    inspect_tools_example()
    error_handling_example()

    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Dynamic tool loading using @module_callback")
    print("2. Meta-tools that return callables to modify the agent")
    print("3. Loading different tool categories on demand")
    print("4. Tools persisting for the rest of execution")
    print("5. Error handling for invalid tool categories")
