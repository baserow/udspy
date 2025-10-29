"""Streaming example with reasoning.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import asyncio

import udspy

# Configure from environment variables
udspy.settings.configure()


# Define a signature with reasoning
class ReasonedQA(udspy.Signature):
    """Answer questions with step-by-step reasoning."""

    question: str = udspy.InputField(description="Question to answer")
    reasoning: str = udspy.OutputField(description="Step-by-step reasoning process")
    answer: str = udspy.OutputField(description="Final answer")


async def main():
    """Run streaming prediction example."""
    predictor = udspy.Predict(ReasonedQA)

    print("Question: What is the sum of the first 10 prime numbers?\n")

    async for item in predictor.astream(question="What is the sum of the first 10 prime numbers?"):
        if isinstance(item, udspy.OutputStreamChunk):
            if not item.is_complete and item.content:
                # Print only the new content (delta)
                if item.delta:
                    print(f"{item.delta}", end="", flush=True)

            if item.is_complete:
                print(f"\n--- {item.field_name} complete ---\n")
        elif isinstance(item, udspy.Prediction):
            print("\n=== Final Result ===")
            print(f"Answer: {item.answer}")


if __name__ == "__main__":
    asyncio.run(main())
