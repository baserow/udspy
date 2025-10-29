"""Example demonstrating History for multi-turn conversations.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import udspy
from udspy import History, InputField, OutputField, Predict, Signature

# Configure from environment variables
udspy.settings.configure()


class QA(Signature):
    """Answer questions with context awareness."""

    question: str = InputField()
    answer: str = OutputField()


def basic_history_example() -> None:
    """Basic multi-turn conversation with History."""
    print("=== Basic History Example ===\n")

    predictor = Predict(QA)
    history = History()

    # First turn
    print("User: What is Python?")
    result = predictor(question="What is Python?", history=history)
    print(f"Assistant: {result.answer}\n")

    # Second turn - context is maintained
    print("User: What are its main features?")
    result = predictor(question="What are its main features?", history=history)
    print(f"Assistant: {result.answer}\n")

    # Third turn - full conversation context
    print("User: Is it good for beginners?")
    result = predictor(question="Is it good for beginners?", history=history)
    print(f"Assistant: {result.answer}\n")

    # View conversation history
    print(f"\n{history}")


def manual_history_management() -> None:
    """Manually manage history for more control."""
    print("\n=== Manual History Management ===\n")

    predictor = Predict(QA)
    history = History()

    # Pre-populate history with context
    history.add_system_message("You are a helpful coding tutor. Keep answers concise.")
    history.add_user_message("I'm learning to code")
    history.add_assistant_message("Great! I'm here to help you learn programming.")

    print(f"Starting with pre-populated history: {len(history)} messages\n")

    # Now ask questions with this context
    print("User: Should I start with Python or JavaScript?")
    result = predictor(question="Should I start with Python or JavaScript?", history=history)
    print(f"Assistant: {result.answer}\n")

    print(f"History now has {len(history)} messages")


def branching_conversations() -> None:
    """Create branching conversations with history.copy()."""
    print("\n=== Branching Conversations ===\n")

    predictor = Predict(QA)
    main_history = History()

    # Start main conversation
    print("User: Tell me about AI")
    result = predictor(question="Tell me about AI", history=main_history)
    print(f"Assistant: {result.answer}\n")

    # Branch 1: Focus on machine learning
    ml_history = main_history.copy()
    print("Branch 1 - User: What is machine learning?")
    result = predictor(question="What is machine learning?", history=ml_history)
    print(f"Assistant: {result.answer}\n")

    # Branch 2: Focus on neural networks
    nn_history = main_history.copy()
    print("Branch 2 - User: What are neural networks?")
    result = predictor(question="What are neural networks?", history=nn_history)
    print(f"Assistant: {result.answer}\n")

    print(f"Main history: {len(main_history)} messages")
    print(f"ML branch: {len(ml_history)} messages")
    print(f"NN branch: {len(nn_history)} messages")


def history_with_clearing() -> None:
    """Use history.clear() to reset conversation."""
    print("\n=== History with Clearing ===\n")

    predictor = Predict(QA)
    history = History()

    # First conversation
    print("Conversation 1:")
    print("User: What is Python?")
    result = predictor(question="What is Python?", history=history)
    print(f"Assistant: {result.answer}")
    print(f"History: {len(history)} messages\n")

    # Clear and start fresh
    history.clear()
    print("History cleared!\n")

    # New conversation - no context from previous
    print("Conversation 2:")
    print("User: What is JavaScript?")
    result = predictor(question="What is JavaScript?", history=history)
    print(f"Assistant: {result.answer}")
    print(f"History: {len(history)} messages")


if __name__ == "__main__":
    # Run examples
    basic_history_example()
    manual_history_management()
    branching_conversations()
    history_with_clearing()
