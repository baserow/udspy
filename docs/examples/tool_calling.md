# Tool Calling

Learn how to use OpenAI's native tool calling with udspy.

## Overview

Tool calling in udspy follows the OpenAI tool calling pattern - a **multi-turn conversation** where the LLM requests tool execution and you provide the results:

```
┌─────────────┐
│   Step 1:   │  You: "What is 157 × 234?"
│  First Call │  LLM: "I need to call Calculator(multiply, 157, 234)"
└─────────────┘
       │
       ▼
┌─────────────┐
│   Step 2:   │  You execute: calculator("multiply", 157, 234) → 36738
│  Execute    │
└─────────────┘
       │
       ▼
┌─────────────┐
│   Step 3:   │  You: "Calculator returned 36738"
│ Second Call │  LLM: "The answer is 36,738"
└─────────────┘
```

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
if result.native_tool_calls:
    print(f"LLM wants to call: {result.native_tool_calls[0].name}")
    print(f"With arguments: {result.native_tool_calls[0].args}")
    # Now you can execute manually or log/analyze the tool calls
```

This is useful for:
- Requiring user approval before executing tools (see confirmation examples)
- Logging or analyzing tool usage patterns
- Implementing custom execution logic
- Rate limiting or caching tool results

## Manual Tool Execution

Define tools as Pydantic models when you want full control:

### 1. Define the Tool Schema

This describes the tool to the LLM - what parameters it takes:

```python
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic operations."""
    operation: str = Field(description="add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
```

### 2. Implement the Tool Function

This is the actual Python code that executes:

```python
def calculator(operation: str, a: float, b: float) -> float:
    """Execute calculator operation."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]
```

### 3. Handle the Multi-Turn Conversation

```python
# First call - LLM decides what to do
predictor = Predict(QA, tools=[Calculator])
result = predictor(question="What is 5 + 3?", auto_execute_tools=False)

if result.native_tool_calls:
    # LLM requested a tool call
    for tool_call in result.native_tool_calls:
        # Execute YOUR implementation
        tool_result = calculator(**tool_call.args)

        # Send result back to LLM (requires manual message construction)
        # See examples/tool_calling_manual.py for complete implementation
```

## Multiple Tools

Provide multiple tools for different operations:

```python
@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(operation: str, a: float, b: float) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

@tool(name="WebSearch", description="Search the web")
def web_search(query: str = Field(description="Search query")) -> str:
    # Your web search implementation
    return f"Search results for: {query}"

@tool(name="DateInfo", description="Get current date/time")
def date_info(timezone: str = Field(description="Timezone name")) -> str:
    # Your date/time implementation
    return f"Current time in {timezone}"

predictor = Predict(
    signature,
    tools=[calculator, web_search, date_info],
)
```

## Key Points

1. **The Schema != The Implementation**
   - Schema (Pydantic model or `@tool` params): Describes the tool to the LLM
   - Implementation (Python function): Your actual code

2. **It's Multi-Turn**
   - Call 1: LLM decides to use a tool
   - You execute the tool
   - Call 2: Send results back to get final answer

3. **You Control Execution**
   - LLM only *requests* tool calls
   - YOU decide if/how to execute them
   - YOU send results back

## Why This Design?

This gives you full control:
- Validate tool calls before executing
- Handle errors gracefully
- Implement tools however you want (API calls, database queries, etc.)
- Add logging, rate limiting, security checks, etc.

The LLM just requests the tool - you're in charge of everything else!

## Complete Examples

See the example files:

- **`tool_calling_auto.py`** - Automatic tool execution with @tool decorator (recommended)
- **`tool_calling_manual.py`** - Manual tool execution with full control
- **`confirmation_loop.py`** - Requiring user approval before tool execution

## Advanced Features

### Async Tools

Tools can be async functions:

```python
@tool(name="AsyncSearch", description="Search with async API")
async def async_search(query: str) -> str:
    # Async implementation
    result = await some_async_api_call(query)
    return result
```

### Tool Confirmation

Require user confirmation before executing certain tools:

```python
@tool(
    name="DeleteFile",
    description="Delete a file",
    require_confirmation=True  # Requires user approval
)
def delete_file(path: str) -> str:
    # Will only execute after user confirmation
    os.remove(path)
    return f"Deleted {path}"
```

See `examples/confirmation_loop.py` for a complete example.
