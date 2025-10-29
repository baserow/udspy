"""Streaming example with reasoning."""

import asyncio
import os

import udspy

# Configure with your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

udspy.settings.configure(api_key=api_key, model="gpt-4o-mini")


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
