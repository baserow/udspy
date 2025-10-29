"""Simple tool calling example showing the complete pattern.

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

import json
from typing import Literal

import udspy
from udspy import InputField, OutputField, Predict, Signature

# Configure from environment variables
udspy.settings.configure()


# Step 1: Implement the actual tool function
@udspy.tool(name="calculator", description="Perform basic arithmetic operations.")
def calculator(
    operation: Literal["add", "subtract", "multiply", "divide"], a: float, b: float
) -> float:
    """Execute calculator operation."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]


# Step 2: Use the OpenAI client directly to handle multi-turn conversation
def ask_with_tools(question: str) -> str:
    """Ask a question with tool support.

    This shows the complete pattern:
    1. First call: LLM decides to use a tool
    2. Execute the tool
    3. Second call: Send tool result back to LLM
    4. LLM provides final answer
    """

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Create predictor with tools
    predictor = Predict(QA, tools=[calculator])

    # Build messages for multi-turn conversation
    messages = [
        {"role": "system", "content": predictor.adapter.format_instructions(predictor.signature)},
        {
            "role": "user",
            "content": predictor.adapter.format_inputs(predictor.signature, {"question": question}),
        },
    ]

    client = udspy.settings.aclient
    tool_schemas = predictor.adapter.format_tool_schemas(predictor.tools.values())

    import asyncio

    # First call - LLM decides what to do
    print("🤖 First call to LLM...")
    response = asyncio.run(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tool_schemas,
        )
    )

    message = response.choices[0].message

    # Check if LLM wants to use tools
    if message.tool_calls:
        print(f"🔧 LLM requested tool: {message.tool_calls[0].function.name}")

        # Add LLM's tool request to conversation
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Execute tool
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        print(f"⚙️  Executing: calculator({args})")

        result = calculator(**args)
        print(f"✅ Tool result: {result}")

        # Add tool result to conversation
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )

        # Second call - LLM uses tool result to answer
        print("🤖 Second call to LLM with tool result...")
        response = asyncio.run(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        )

        return response.choices[0].message.content or "No answer"

    else:
        # LLM answered directly without tools
        print("💬 LLM answered directly (no tools needed)")
        return message.content or "No answer"


if __name__ == "__main__":
    question = "What is 157 multiplied by 234?"
    print(f"Question: {question}\n")

    answer = ask_with_tools(question)

    print(f"\n📝 Final Answer: {answer}")
