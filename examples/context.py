"""Example demonstrating context-specific settings.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import os

import udspy
from udspy import InputField, OutputField, Predict, Signature
from udspy.lm import LM

# Configure global settings from environment variables
udspy.settings.configure()


class QA(Signature):
    """Answer questions concisely."""

    question: str = InputField()
    answer: str = OutputField()


if __name__ == "__main__":
    predictor = Predict(QA)

    # Use global settings (gpt-4o-mini)
    print("=== Using global settings (gpt-4o-mini) ===")
    result = predictor(question="What is 2+2?")
    print(f"Answer: {result.answer}\n")

    # Temporarily use a different model in a specific context
    print("=== Using context-specific model (gpt-4) ===")
    api_key = os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
    context_lm = LM(model="gpt-4", api_key=api_key)
    with udspy.settings.context(lm=context_lm, temperature=0.0):
        result = predictor(question="What is the capital of France?")
        print(f"Answer: {result.answer}")
        print("Model used: gpt-4\n")

    # Back to global settings
    print("=== Back to global settings ===")
    result = predictor(question="What is Python?")
    print(f"Answer: {result.answer}")
    print("Model used: gpt-4o-mini\n")

    # Example: Using different API keys for different tenants/users
    print("=== Multi-tenant example ===")

    # Simulate different users with different API keys
    # Falls back to the global API key if user-specific keys are not set
    global_api_key = os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
    user1_api_key = os.getenv("USER1_API_KEY", global_api_key)
    user2_api_key = os.getenv("USER2_API_KEY", global_api_key)

    print("User 1 request:")
    user1_lm = LM(model="gpt-4o-mini", api_key=user1_api_key)
    with udspy.settings.context(lm=user1_lm):
        result = predictor(question="What is AI?")
        print(f"Answer: {result.answer}\n")

    print("User 2 request:")
    user2_lm = LM(model="gpt-4o-mini", api_key=user2_api_key)
    with udspy.settings.context(lm=user2_lm):
        result = predictor(question="What is ML?")
        print(f"Answer: {result.answer}\n")

    # Nested contexts
    print("=== Nested contexts ===")
    outer_lm = LM(model="gpt-4", api_key=api_key)
    with udspy.settings.context(lm=outer_lm, temperature=0.5):
        print("Outer context (gpt-4, temp=0.5)")

        # Inner context only changes temperature, keeps same LM
        with udspy.settings.context(temperature=0.9):
            print("Inner context (gpt-4, temp=0.9)")
            # This will use gpt-4 with temperature=0.9

        print("Back to outer context (gpt-4, temp=0.5)")
