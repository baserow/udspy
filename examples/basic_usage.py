"""Basic usage example of udspy."""

import os

import udspy
from udspy import InputField, OutputField, Predict, Signature

# Configure with your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

udspy.settings.configure(api_key=api_key, model="gpt-4o-mini")


# Define a simple question-answering signature
class QA(Signature):
    """Answer questions concisely and accurately."""

    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")


# Create a predictor
predictor = Predict(QA)

# Make predictions
if __name__ == "__main__":
    result = predictor(question="What is the capital of France?")
    print(f"Question: What is the capital of France?")
    print(f"Answer: {result.answer}")

    result = predictor(question="What is 15 * 23?")
    print(f"\nQuestion: What is 15 * 23?")
    print(f"Answer: {result.answer}")
