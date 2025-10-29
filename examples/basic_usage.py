"""Basic usage example of udspy.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import udspy
from udspy import InputField, OutputField, Predict, Signature

# Configure from environment variables (UDSPY_LM_API_KEY, UDSPY_LM_MODEL)
# Falls back to OPENAI_API_KEY if UDSPY_LM_API_KEY is not set
udspy.settings.configure()


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
    print("Question: What is the capital of France?")
    print(f"Answer: {result.answer}")

    result = predictor(question="What is 15 * 23?")
    print("\nQuestion: What is 15 * 23?")
    print(f"Answer: {result.answer}")
