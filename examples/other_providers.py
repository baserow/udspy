"""Example showing how to use udspy with different LLM providers.

This example demonstrates using OpenAI, Groq, and Ollama with the same code
by switching providers using model prefixes or context managers.

Before running, set provider-specific API keys:
    export OPENAI_API_KEY="sk-..."
    export GROQ_API_KEY="gsk-..."
    # No API key needed for Ollama (local)
"""

import os
from random import randint

import udspy
from udspy import InputField, OutputField, Predict, Signature


class QA(Signature):
    """Answer questions concisely and accurately."""

    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")


if __name__ == "__main__":
    # Example 1: OpenAI (default provider)
    print("\n=== OpenAI ===")
    with udspy.settings.context(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")):
        predictor = Predict(QA)
        result = predictor(question="What is the capital of France?")
        print(f"Answer: {result.answer}")

    # Example 2: Groq (using model prefix)
    print("\n=== Groq (with prefix) ===")
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        with udspy.settings.context(model="groq/llama-3-70b", api_key=groq_key):
            predictor = Predict(QA)
            result = predictor(question="What is 15 * 23?")
            print(f"Answer: {result.answer}")
    else:
        print("GROQ_API_KEY not set, skipping Groq example")

    # Example 3: Groq (using explicit base_url)
    print("\n=== Groq (with base_url) ===")
    if groq_key:
        with udspy.settings.context(
            model="llama-3-70b",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        ):
            predictor = Predict(QA)
            result = predictor(question=f"What is {randint(1, 100)} + {randint(1, 100)}?")
            print(f"Answer: {result.answer}")
    else:
        print("GROQ_API_KEY not set, skipping Groq example")

    # Example 4: Ollama (local, using model prefix - no API key needed)
    print("\n=== Ollama (with prefix) ===")
    try:
        with udspy.settings.context(model="ollama/llama2"):
            predictor = Predict(QA)
            result = predictor(question="What is Python?")
            print(f"Answer: {result.answer}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")

    # Example 5: Ollama (using explicit base_url)
    print("\n=== Ollama (with base_url) ===")
    try:
        with udspy.settings.context(model="llama2", base_url="http://localhost:11434/v1"):
            predictor = Predict(QA)
            result = predictor(question="What is TypeScript?")
            print(f"Answer: {result.answer}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")
