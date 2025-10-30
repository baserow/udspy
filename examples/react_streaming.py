"""ReAct streaming example.

This example demonstrates how to stream the ReAct agent's reasoning and tool usage
in real-time, allowing you to see the agent's thought process as it happens.
"""

import asyncio

from pydantic import Field

from udspy import InputField, OutputField, ReAct, Signature, settings, tool
from udspy.lm import LM
from udspy.streaming import OutputStreamChunk, Prediction, ThoughtStreamChunk

lm = LM(model="gpt-oss:120b-cloud", base_url="http://localhost:11434/v1")
settings.configure(lm=lm)


# Define tools for the agent
@tool(name="search", description="Search for information on the internet")
def search(query: str = Field(description="The search query")) -> str:
    """Mock search tool - in a real application, this would call a search API."""
    search_results = {
        "python": "Python is a high-level, interpreted programming language known for its readability and versatility. Created by Guido van Rossum and first released in 1991.",
        "paris": "Paris weather: Partly cloudy, 18°C (64°F), humidity 65%. Light rain expected tomorrow.",
        "tokyo": "Tokyo has a population of approximately 14 million people in the city proper, and about 37 million in the greater Tokyo area.",
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
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# Define the task signature
class ResearchTask(Signature):
    """Research a topic and provide a comprehensive answer using available tools."""

    question: str = InputField(description="The research question or task")
    answer: str = OutputField(description="Comprehensive answer based on tool usage")


async def stream(agent: ReAct, question: str):
    """Helper function to stream agent events."""

    current_field = None
    final_result = None
    thought = None

    async for event in agent.astream(question=question):
        if isinstance(event, ThoughtStreamChunk):
            # Print thought process as it streams
            if thought is None:
                print("\n[THOUGHT PROCESS]")
                thought = event
            if event.delta and not event.is_complete:
                print(f"{event.delta}", end="", flush=True)
            elif event.is_complete:
                print("[END OF THOUGHT PROCESS]\n\n")  # End of thought line

        elif isinstance(event, OutputStreamChunk):
            # New field started
            if event.field_name != current_field:
                if current_field is not None:
                    print()  # End previous field
                current_field = event.field_name
                print(f"\n[{event.field_name.upper()}]")

            # Print streaming content as it arrives
            if event.delta and not event.is_complete:
                print(event.delta, end="", flush=True)

        elif isinstance(event, Prediction):
            # Store final result (last Prediction is the final one)
            final_result = event

    return final_result


async def research_example_1():
    # Create ReAct agent with tools
    agent = ReAct(
        ResearchTask,
        tools=[search, calculator],
        enable_ask_to_user=False,
        max_iters=5,
    )

    question = "What is Python and how many letters are in the word 'Python'?"
    print(f"Question: {question}\n")
    print("Agent is thinking...\n")
    print("-" * 80)

    # Stream the agent's reasoning process in real-time
    final_result = await stream(agent, question)

    print("\n" + "-" * 80)
    print("\n=== Final Result ===")

    # Check if answer field exists
    if hasattr(final_result, "answer") and final_result.answer:
        print(f"\nAnswer: {final_result.answer}\n")
    else:
        print("\n(Note: No final answer extracted)\n")


async def streaming_example_2():
    agent = ReAct(
        ResearchTask,
        tools=[search, calculator],
        enable_ask_to_user=False,
        max_iters=5,
    )

    question = (
        "Search for Tokyo population and calculate 37 million divided by 1000. "
        "Once you have both pieces of information, return the results ordered."
    )

    print(f"Question: {question}\n")
    print("Agent is thinking...\n")
    print("-" * 80)

    result = await stream(agent, question)
    print("\n" + "-" * 80)
    print("\n=== Final Result ===")

    if hasattr(result, "answer") and result.answer:
        print(f"✓ Answer:\n{result.answer}\n")
    else:
        print("✓ Completed (no final answer extracted)\n")


async def main():
    """Run all streaming examples."""

    await research_example_1()
    await streaming_example_2()

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
