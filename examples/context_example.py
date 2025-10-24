"""Example demonstrating context-specific settings."""

import os

import udspy
from udspy import InputField, OutputField, Predict, Signature

# Configure global settings
global_api_key = os.getenv("OPENAI_API_KEY")
if not global_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

udspy.settings.configure(api_key=global_api_key, model="gpt-4o-mini")


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
    with udspy.settings.context(model="gpt-4", temperature=0.0):
        result = predictor(question="What is the capital of France?")
        print(f"Answer: {result.answer}")
        print(f"Model used: gpt-4\n")

    # Back to global settings
    print("=== Back to global settings ===")
    result = predictor(question="What is Python?")
    print(f"Answer: {result.answer}")
    print(f"Model used: gpt-4o-mini\n")

    # Example: Using different API keys for different tenants/users
    print("=== Multi-tenant example ===")

    # Simulate different users with different API keys
    user1_api_key = os.getenv("USER1_API_KEY", global_api_key)
    user2_api_key = os.getenv("USER2_API_KEY", global_api_key)

    print("User 1 request:")
    with udspy.settings.context(api_key=user1_api_key):
        result = predictor(question="What is AI?")
        print(f"Answer: {result.answer}\n")

    print("User 2 request:")
    with udspy.settings.context(api_key=user2_api_key):
        result = predictor(question="What is ML?")
        print(f"Answer: {result.answer}\n")

    # Nested contexts
    print("=== Nested contexts ===")
    with udspy.settings.context(model="gpt-4", temperature=0.5):
        print("Outer context (gpt-4, temp=0.5)")

        with udspy.settings.context(temperature=0.9):
            print("Inner context (gpt-4, temp=0.9)")
            # This will use gpt-4 with temperature=0.9

        print("Back to outer context (gpt-4, temp=0.5)")
