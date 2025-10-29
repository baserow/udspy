"""Example showing how to use udspy with different LLM providers.

This example demonstrates using OpenAI, Groq, AWS Bedrock, Anthropic, and Ollama
with the same code by switching providers using context managers.

Before running, optionally set provider-specific API keys:
    export OPENAI_API_KEY="sk-..."
    export GROQ_API_KEY="..."
    export AWS_BEDROCK_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
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
    for settings in [
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
        },
        {
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": "openai/gpt-oss-20b",
            "base_url": "https://api.groq.com/openai/v1",
        },
        {
            "api_key": os.getenv("AWS_BEDROCK_API_KEY"),
            "model": "openai.gpt-oss-20b-1:0",
            "base_url": "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/",
        },
        {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-haiku-4-5",
            "base_url": "https://api.anthropic.com/v1/",
        },
        {
            "model": "gpt-oss:120b-cloud",
            "base_url": "http://localhost:11434/v1",  # Ollama local LLM server
        },
    ]:
        with udspy.settings.context(**settings):
            predictor = Predict(QA)

            print(f"\nModel: {settings.get('model')}")
            try:
                question = "What is the capital of France?"
                result = predictor(question=question)
                print(f"Question: {question}")
                print(f"Answer from : {result.answer}")

                question = f"What is {randint(1, 100)} * {randint(1, 100)}?"
                result = predictor(question=question)
                print(f"\nQuestion: {question}")
                print(f"Answer: {result.answer}")
            except Exception as e:
                print(f"Error during prediction with settings {settings}: {e}")
