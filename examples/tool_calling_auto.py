"""Example with native tool calling - automatic execution with @tool decorator.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

from pydantic import Field

import udspy
from udspy import InputField, OutputField, Predict, Signature, tool

# Configure from environment variables
udspy.settings.configure()


# Define tools using the @tool decorator
@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="The operation: add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Execute calculator operation."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]


@tool(name="WebSearch", description="Search the web for information")
def web_search(query: str = Field(description="Search query")) -> str:
    """Execute web search (mock implementation)."""
    return f"Search results for: {query}"


# Define signature
class MathQuery(Signature):
    """Answer math questions using available tools."""

    question: str = InputField(description="Math question")
    answer: str = OutputField(description="Answer to the question")


if __name__ == "__main__":
    # Create predictor with tool callables
    # The @tool decorator makes them executable, so Predict handles everything automatically!
    predictor = Predict(MathQuery, tools=[calculator, web_search], max_turns=5)

    # Just call it! The multi-turn loop is handled automatically
    question = "What is 157 multiplied by 234?"
    print(f"Question: {question}\n")

    result = predictor(question=question)

    print("\n=== Final Answer ===")
    print(f"Answer: {result.answer}")
