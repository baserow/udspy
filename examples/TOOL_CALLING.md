# Tool Calling Guide

Tool calling in udspy follows the OpenAI tool calling pattern. Here's how it works:

## The Pattern

Tool calling is a **multi-turn conversation**:

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

## Three Steps to Implement

### 1. Define the Tool Schema (Pydantic Model)

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
result = predictor(question="What is 5 + 3?")

if "tool_calls" in result:
    # LLM requested a tool call
    for tool_call in result.tool_calls:
        # Parse the arguments
        args = json.loads(tool_call["arguments"])

        # Execute YOUR implementation
        tool_result = calculator(**args)

        # Send result back to LLM (second call)
        # Build messages with tool result and call LLM again
        ...
```

## Complete Examples

See the example files:

- **`tool_calling_manual.py`** - Clear, step-by-step example with annotations
- **`tool_calling_auto.py`** - Complete implementation with error handling

## Key Points

1. **The Schema != The Implementation**
   - Schema (Pydantic model): Describes the tool to the LLM
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
