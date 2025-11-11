"""Example showing how to distinguish predictions from different modules during streaming.

This example demonstrates:
1. How to identify which module produced a prediction (parent vs child)
2. How to distinguish final predictions from intermediate results
3. How to handle streaming from nested modules (ReAct with ChainOfThought)
"""

import asyncio
import math
import os

from openai import AsyncOpenAI

import udspy
from udspy import InputField, OpenAILM, OutputField, ReAct, Signature, tool
from udspy.streaming import Prediction


class QA(Signature):
    """Answer questions with reasoning."""

    question: str = InputField()
    answer: str = OutputField()


@tool(name="calculator", description="Perform calculations")
def calculator(expression: str) -> str:
    """Simple calculator that evaluates math expressions."""
    try:
        # Safe evaluation - only allows basic math
        result = eval(expression, {"__builtins__": {"sqrt": math.sqrt}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


async def main():
    # Configure udspy
    api_key = os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL")
    model = os.getenv("UDSPY_LM_MODEL", "gpt-4o-mini")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    lm = OpenAILM(client=client, default_model=model)
    udspy.settings.configure(lm=lm)

    # Create a ReAct agent with ChainOfThought for extraction
    # ReAct will use Predict internally, and we use ChainOfThought for the extraction step
    agent = ReAct(QA, tools=[calculator])

    print("=" * 80)
    print("Streaming with Module Information")
    print("=" * 80)
    print()

    question = "What is sqrt(42)?"

    print(f"Question: {question}\n")
    print("Streaming events:")
    print("-" * 80)

    # Stream and distinguish predictions from different modules
    async for event in agent.astream(question=question):
        if isinstance(event, Prediction):
            # Show which module produced the prediction
            module_name = event.module.__class__.__name__ if event.module else "Unknown"
            is_final = "FINAL" if event.is_final else "intermediate"

            print(f"\n[{module_name}] Prediction ({is_final}):")

            # Show the prediction fields
            for key, value in event.items():
                if key not in ("module", "is_final", "native_tool_calls", "trajectory"):
                    print(f"  {key}: {value}")

            # Only process the final prediction
            if event.is_final and event.module is agent:
                print("\n" + "=" * 80)
                print("FINAL ANSWER:")
                print(f"  {event.answer}")
                print("=" * 80)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
