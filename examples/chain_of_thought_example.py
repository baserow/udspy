"""Chain of Thought reasoning example."""

import os

import udspy
from udspy import ChainOfThought, InputField, OutputField, Signature

# Configure with your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

udspy.settings.configure(api_key=api_key, model="gpt-4o-mini")


# Define a simple question-answering signature
class QA(Signature):
    """Answer questions with clear reasoning."""

    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Final answer")


# Create a Chain of Thought predictor
# This automatically adds a "reasoning" field before the answer
cot_predictor = ChainOfThought(QA)


if __name__ == "__main__":
    # Example 1: Simple math question
    print("=" * 60)
    print("Example 1: Math Question")
    print("=" * 60)

    result = cot_predictor(question="What is 15 * 23?")

    print("Question: What is 15 * 23?")
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nAnswer: {result.answer}")

    # Example 2: Logic puzzle
    print("\n" + "=" * 60)
    print("Example 2: Logic Puzzle")
    print("=" * 60)

    result = cot_predictor(
        question="If all roses are flowers and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?"
    )

    print(
        "Question: If all roses are flowers and some flowers fade quickly, "
        "can we conclude that some roses fade quickly?"
    )
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nAnswer: {result.answer}")

    # Example 3: Word problem
    print("\n" + "=" * 60)
    print("Example 3: Word Problem")
    print("=" * 60)

    result = cot_predictor(
        question="A train travels 60 miles in 1 hour. "
        "How far will it travel in 2.5 hours at the same speed?"
    )

    print(
        "Question: A train travels 60 miles in 1 hour. "
        "How far will it travel in 2.5 hours at the same speed?"
    )
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nAnswer: {result.answer}")

    # Example 4: Custom reasoning description
    print("\n" + "=" * 60)
    print("Example 4: Custom Reasoning Description")
    print("=" * 60)

    custom_cot = ChainOfThought(
        QA,
        reasoning_description="Detailed step-by-step analysis with intermediate calculations",
    )

    result = custom_cot(question="What is the sum of the first 5 prime numbers?")

    print("Question: What is the sum of the first 5 prime numbers?")
    print(f"\nDetailed Analysis:\n{result.reasoning}")
    print(f"\nAnswer: {result.answer}")

    # Example 5: Complex multi-step reasoning
    print("\n" + "=" * 60)
    print("Example 5: Complex Multi-Step Problem")
    print("=" * 60)

    class Analysis(Signature):
        """Analyze scenarios and provide recommendations."""

        scenario: str = InputField(description="Scenario to analyze")
        analysis: str = OutputField(description="Detailed analysis")
        recommendation: str = OutputField(description="Recommended action")

    analyzer = ChainOfThought(Analysis)

    result = analyzer(
        scenario="A startup has $100k runway, 6 months left, "
        "and needs to choose between hiring a salesperson or "
        "building a new feature."
    )

    print(
        "Scenario: A startup has $100k runway, 6 months left, "
        "and needs to choose between hiring a salesperson or "
        "building a new feature."
    )
    print(f"\nReasoning:\n{result.reasoning}")
    print(f"\nAnalysis:\n{result.analysis}")
    print(f"\nRecommendation:\n{result.recommendation}")
