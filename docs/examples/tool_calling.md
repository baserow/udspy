# Tool Calling

Learn how to use OpenAI's native tool calling with udspy.

## Two Ways to Use Tools

udspy supports two approaches to tool calling:

1. **Automatic Execution with `@tool` decorator** (Recommended) - Tools are automatically executed
2. **Manual Execution with Pydantic models** - You handle tool execution yourself

## Automatic Tool Execution (Recommended)

Use the `@tool` decorator to mark functions as executable tools. udspy will automatically execute them and handle multi-turn conversations:

```python
from pydantic import Field
from udspy import tool, Predict, Signature, InputField, OutputField

@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, or divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Execute calculator operation."""
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

class MathQuery(Signature):
    """Answer math questions."""
    question: str = InputField()
    answer: str = OutputField()

# Tools decorated with @tool are automatically executed
predictor = Predict(MathQuery, tools=[calculator])
result = predictor(question="What is 157 times 234?")
print(result.answer)  # "The answer is 36738"
```

The predictor automatically:
1. Detects when the LLM wants to call a tool
2. Executes the tool function
3. Sends the result back to the LLM
4. Returns the final answer

## Optional Tool Execution

You can control whether tools are automatically executed:

```python
# Default: auto_execute_tools=True
result = predictor(question="What is 5 + 3?")
print(result.answer)  # "The answer is 8"

# Get tool calls without execution
result = predictor(question="What is 5 + 3?", auto_execute_tools=False)
if "tool_calls" in result:
    print(f"LLM wants to call: {result.tool_calls[0]['name']}")
    print(f"With arguments: {result.tool_calls[0]['arguments']}")
    # Now you can execute manually or log/analyze the tool calls
```

This is useful for:
- Requiring user approval before executing tools
- Logging or analyzing tool usage patterns
- Implementing custom execution logic
- Rate limiting or caching tool results

## Manual Tool Execution

Define tools as Pydantic models when you want full control:

```python
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic operations."""
    operation: str = Field(description="add, subtract, multiply, or divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

# Pydantic models are schema-only (not automatically executed)
predictor = Predict(MathQuery, tools=[Calculator])
result = predictor(question="What is 157 times 234?")

# You must check for and execute tool calls yourself
if "tool_calls" in result:
    for tool_call in result.tool_calls:
        print(f"Called: {tool_call['name']}")
        print(f"Arguments: {tool_call['arguments']}")
        # Execute manually and construct follow-up messages
```

## Multiple Tools

Provide multiple tools for different operations:

```python
class Calculator(BaseModel):
    """Perform arithmetic operations."""
    operation: str
    a: float
    b: float

class WebSearch(BaseModel):
    """Search the web."""
    query: str = Field(description="Search query")

class DateInfo(BaseModel):
    """Get date information."""
    timezone: str = Field(description="Timezone name")

predictor = Predict(
    signature,
    tools=[Calculator, WebSearch, DateInfo],
)
```

## Tool Execution

Execute tool calls and continue the conversation:

```python
def execute_calculator(operation: str, a: float, b: float) -> float:
    """Execute calculator tool."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    return ops[operation](a, b)

# Get initial response with tool calls
result = predictor(question="What is 15 * 23?")

if "tool_calls" in result:
    # Execute tool calls
    tool_results = []
    for tool_call in result.tool_calls:
        if tool_call["name"] == "Calculator":
            args = json.loads(tool_call["arguments"])
            result_value = execute_calculator(**args)
            tool_results.append({
                "id": tool_call["id"],
                "result": result_value,
            })

    # Continue conversation with tool results
    # (requires manual message construction - see advanced examples)
```

See the [full example](https://github.com/silvestrid/udspy/blob/main/examples/tool_calling.py) in the repository.
