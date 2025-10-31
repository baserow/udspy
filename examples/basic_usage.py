"""Basic usage example of udspy.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import os

import udspy
from udspy import InputField, OutputField, OpenAILM, Predict, Signature
from openai import AsyncOpenAI

if __name__ == "__main__":
    # APPROACH 1: Direct LM usage (simplest)
    print("=== Direct LM Usage ===\n")

    api_key = os.getenv("UDSPY_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("UDSPY_LM_BASE_URL")
    model = os.getenv("UDSPY_LM_MODEL", "gpt-4o-mini")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    lm = OpenAILM(client=client, default_model=model)

    answer = lm("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {answer}\n")

    answer = lm("What is 15 * 23?")
    print(f"Q: What is 15 * 23?")
    print(f"A: {answer}\n")

    # APPROACH 2: Structured signatures (for complex outputs)
    print("=== Structured Signatures ===\n")

    udspy.settings.configure()

    class QA(Signature):
        """Answer questions concisely and accurately."""

        question: str = InputField(description="Question to answer")
        answer: str = OutputField(description="Concise answer")

    predictor = Predict(QA)

    result = predictor(question="What is the capital of Germany?")
    print(f"Q: What is the capital of Germany?")
    print(f"A: {result.answer}\n")
