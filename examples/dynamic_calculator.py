"""Example demonstrating dynamic tool loading with a calculator.

This example shows a ReAct agent that starts without a calculator tool,
recognizes it needs one, loads it dynamically, and then uses it to solve
a math problem. The trajectory clearly shows:
1. User makes a request requiring calculation
2. Agent loads calculator tool via module callback
3. Agent uses the newly loaded calculator in the next step

Before running, set environment variables:
    export UDSPY_LM_API_KEY="sk-..."  # or OPENAI_API_KEY
    export UDSPY_LM_MODEL="gpt-4o-mini"
"""

from pydantic import Field

import udspy
from udspy import (
    InputField,
    OutputField,
    ReAct,
    Signature,
    module_callback,
    tool,
)


@tool(name="calculator", description="Perform mathematical calculations")
def calculator(expression: str = Field(..., description="Math expression to evaluate")) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Math expression like "15 * 23" or "(100 + 50) / 3"

    Returns:
        String with the result
    """
    
    print(f"[Calculator Tool] Evaluating expression: {expression}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool(name="load_calculator", description="Load the calculator tool for math operations")
def load_calculator() -> callable:  # type: ignore[valid-syntax]
    """Load the calculator tool dynamically.

    This tool returns a module callback that adds the calculator tool
    to the agent's available tools.

    Returns:
        Module callback that loads calculator
    """

    @module_callback
    def add_calculator(context):
        current_tools = [
            t for t in context.module.tools.values() if t.name not in ("finish", "ask_to_user")
        ]
        
        print("[Module Callback] Loading calculator tool...")

        context.module.init_module(tools=current_tools + [calculator])

        return "Calculator tool loaded successfully. You can now use it for calculations."

    return add_calculator


class Question(Signature):
    """Answer questions. Load tools if needed for the task."""

    question: str = InputField()
    answer: str = OutputField()


def single_calculation_example() -> None:
    """Run calculator example showing dynamic tool loading."""
    print("=== Dynamic Calculator Example ===\n")

    agent = ReAct(Question, tools=[load_calculator], max_iters=5)

    print("Initial tools available:")
    for tool_name in agent.tools:
        print(f"  - {tool_name}")
    print()

    question = "Use the calculator to evaluate 15.7 multiplied by 834?"
    print(f"Question: {question}\n")

    result = agent(question=question)

    print(f"\n{'='*60}")
    print(f"Final Answer: {result.answer}")
    print(f"{'='*60}\n")

    print("Tools available after execution:")
    for tool_name in agent.tools:
        print(f"  - {tool_name}")
    print()

    print("\nTrajectory breakdown:")
    print("Step 1: Agent recognizes it needs a calculator")
    print("Step 2: Agent calls 'load_calculator' tool")
    print("Step 3: Module callback adds calculator to available tools")
    print("Step 4: Agent calls 'calculator' tool with '157 * 834'")
    print("Step 5: Agent calls 'finish' with the result")


def multi_calculation_example() -> None:
    """Example with multiple calculations."""
    print("\n=== Multiple Calculations Example ===\n")

    agent = ReAct(Question, tools=[load_calculator], max_iters=8)

    question = "Use the calculator to evaluate (25 + 75) divided by 4, and then multiplied by 12?"
    print(f"Question: {question}\n")

    result = agent(question=question)

    print(f"\n{'='*60}")
    print(f"Final Answer: {result.answer}")
    print(f"{'='*60}\n")


def conditional_loading_example() -> None:
    """Example showing calculator is only loaded when needed."""
    print("\n=== Conditional Loading Example ===\n")

    agent = ReAct(Question, tools=[load_calculator], max_iters=5)

    question = "What is the capital of France?"
    print(f"Question: {question}\n")

    result = agent(question=question)

    print(f"\n{'='*60}")
    print(f"Final Answer: {result.answer}")
    print(f"{'='*60}\n")

    print("Note: Calculator was NOT loaded because it wasn't needed")


if __name__ == "__main__":
    udspy.settings.configure()

    single_calculation_example()
    multi_calculation_example()
    conditional_loading_example()

    print("\n=== Key Concepts Demonstrated ===")
    print("1. Agent starts without calculator tool")
    print("2. Agent recognizes need and loads calculator dynamically")
    print("3. Calculator becomes available in the SAME execution")
    print("4. Agent uses newly loaded tool in subsequent trajectory steps")
    print("5. Tools only loaded when actually needed")
