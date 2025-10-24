# Tool Calling Examples

Learn how to use OpenAI's native tool calling with udspy.

## Basic Tool Calling

Define tools as Pydantic models:

```python
from pydantic import BaseModel, Field
from udspy import Predict, Signature, InputField, OutputField

class Calculator(BaseModel):
    """Perform arithmetic operations."""
    operation: str = Field(description="add, subtract, multiply, or divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class MathQuery(Signature):
    """Answer math questions."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(MathQuery, tools=[Calculator])
result = predictor(question="What is 157 times 234?")
print(result.answer)

# Check if tools were called
if "tool_calls" in result:
    for tool_call in result.tool_calls:
        print(f"Called: {tool_call['name']}")
        print(f"Arguments: {tool_call['arguments']}")
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

See the [full example](../../examples/tool_calling_example.py) in the repository.
