"""Example with native tool calling."""

import os

from pydantic import BaseModel, Field

import udspy
from udspy import InputField, OutputField, Predict, Signature

# Configure with your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

udspy.settings.configure(api_key=api_key, model="gpt-4o-mini")


# Define tool schemas using Pydantic
class Calculator(BaseModel):
    """Perform arithmetic operations."""

    operation: str = Field(description="The operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class WebSearch(BaseModel):
    """Search the web for information."""

    query: str = Field(description="Search query")


# Define signature
class MathQuery(Signature):
    """Answer math questions using available tools."""

    question: str = InputField(description="Math question")
    answer: str = OutputField(description="Answer to the question")


# Create predictor with tools
predictor = Predict(MathQuery, tools=[Calculator, WebSearch])

if __name__ == "__main__":
    result = predictor(question="What is 157 multiplied by 234?")

    print(f"Question: What is 157 multiplied by 234?")
    print(f"Answer: {result.answer}")

    # Check if tools were called
    if "tool_calls" in result:
        print("\nTool calls made:")
        for tool_call in result.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['arguments']}")
