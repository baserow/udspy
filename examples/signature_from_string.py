"""Example: Creating signatures from strings (DSPy-style).

This example shows how to use Signature.from_string() to create signatures
using the DSPy-style string format "inputs -> outputs".
"""

import asyncio

import udspy
from udspy import Predict, Signature
from udspy.lm import LM

# Configure API
lm = LM(model="gpt-4o-mini", api_key="your-api-key")
udspy.settings.configure(lm=lm)


def basic_example():
    """Basic signature from string."""
    print("=" * 60)
    print("Basic Example: Simple QA")
    print("=" * 60)

    # DSPy-style string format
    QA = Signature.from_string("question -> answer", "Answer questions concisely")

    predictor = Predict(QA)
    result = predictor(question="What is the capital of France?")

    print("Question: What is the capital of France?")
    print(f"Answer: {result.answer}")
    print()


def multiple_inputs_example():
    """Signature with multiple inputs."""
    print("=" * 60)
    print("Multiple Inputs Example")
    print("=" * 60)

    # Multiple inputs separated by commas
    ContextQA = Signature.from_string(
        "context, question -> answer", "Answer questions using provided context"
    )

    predictor = Predict(ContextQA)
    result = predictor(
        context="Python is a high-level programming language known for its simplicity.",
        question="What is Python known for?",
    )

    print("Question: What is Python known for?")
    print(f"Answer: {result.answer}")
    print()


def multiple_outputs_example():
    """Signature with multiple outputs."""
    print("=" * 60)
    print("Multiple Outputs Example")
    print("=" * 60)

    # Multiple outputs separated by commas
    Analyze = Signature.from_string(
        "text -> summary, sentiment, keywords", "Analyze text comprehensively"
    )

    predictor = Predict(Analyze)
    result = predictor(text="I absolutely love this product! It's amazing and works perfectly.")

    print("Text: I absolutely love this product! It's amazing and works perfectly.")
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Keywords: {result.keywords}")
    print()


def comparison_example():
    """Compare string format vs class-based format."""
    print("=" * 60)
    print("Comparison: String vs Class-based")
    print("=" * 60)

    # String format (simple, all fields are str)
    _ = Signature.from_string("question -> answer")

    # Class-based format (more control)
    class QA_Class(Signature):  # noqa: ARG001
        """Answer questions."""

        question: str = udspy.InputField(description="Question to answer")
        answer: str = udspy.OutputField(description="Concise answer")

    print("String format: Signature.from_string('question -> answer')")
    print("  - Quick and simple")
    print("  - All fields default to str")
    print("  - No field descriptions")
    print()

    print("Class-based format:")
    print("  - Full control over types")
    print("  - Field descriptions")
    print("  - Better for complex signatures")
    print()


async def async_example():
    """Use string signatures with async."""
    print("=" * 60)
    print("Async Example")
    print("=" * 60)

    QA = Signature.from_string("question -> answer", "Answer questions")
    predictor = Predict(QA)

    result = await predictor.aforward(question="What is async/await in Python?")
    print("Question: What is async/await in Python?")
    print(f"Answer: {result.answer}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Signature.from_string() Examples")
    print("DSPy-style signature creation")
    print("=" * 60 + "\n")

    basic_example()
    multiple_inputs_example()
    multiple_outputs_example()
    comparison_example()
    asyncio.run(async_example())

    print("=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. Use Signature.from_string() for quick, simple signatures")
    print("2. Format: 'input1, input2 -> output1, output2'")
    print("3. All fields default to str type")
    print("4. For complex signatures with types/descriptions, use classes")
    print("5. DSPy-compatible API for easy migration")
    print()


if __name__ == "__main__":
    main()
