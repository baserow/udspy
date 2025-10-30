# Predict Module

The `Predict` module is the foundational building block of udspy. It takes a signature and generates outputs from inputs using an LLM.

## Overview

`Predict` is the simplest and most essential module in udspy. It:

- Maps signature inputs to outputs via an LLM
- Supports native tool calling for function execution
- Handles streaming for real-time output generation
- Manages conversation history and message formatting

## Basic Usage

```python
from udspy import Predict, Signature, InputField, OutputField

class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)
result = predictor(question="What is Python?")
print(result.answer)
```

## String Signatures

For quick prototyping, use string signatures:

```python
predictor = Predict("question -> answer")
result = predictor(question="What is Python?")
```

See [Signatures](../signatures.md) for more details.

## Configuration

### Model Selection

```python
# Global default
import udspy
from udspy import LM

lm = LM(model="gpt-4o-mini", api_key="sk-...")
udspy.settings.configure(lm=lm)

# Per-module override
predictor = Predict(QA, model="gpt-4o")
```

### Temperature and Sampling

```python
predictor = Predict(
    QA,
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
)
```

### Custom Adapter

```python
from udspy import ChatAdapter

adapter = ChatAdapter()
predictor = Predict(QA, adapter=adapter)
```

## Tool Calling

Predict supports native OpenAI tool calling:

```python
from udspy import tool
from pydantic import Field

@tool(name="search", description="Search for information")
def search(query: str = Field(...)) -> str:
    return f"Results for: {query}"

predictor = Predict(QA, tools=[search])
result = predictor(question="What is the weather in Tokyo?")

# Access tool calls
for call in result.tool_calls:
    print(f"Called {call['name']} with {call['arguments']}")
```

### Auto-execution vs Manual

By default, tools are NOT auto-executed. You control execution:

```python
# Auto-execute tools
result = await predictor.aexecute(
    question="Search for Python",
    auto_execute_tools=True
)

# Manual execution
result = await predictor.aexecute(
    question="Search for Python",
    auto_execute_tools=False
)
# Tools are returned in result.tool_calls but not executed
```

See [Tool Calling](../../api/tool.md) for more details.

## Streaming

Stream outputs in real-time:

```python
async for event in predictor.aexecute(
    stream=True,
    question="Explain quantum computing"
):
    if isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\n\nFinal: {event.answer}")
```

### Stream Events

The streaming API yields:
- `StreamChunk(field: str, delta: str)` - Incremental text for a field
- `Prediction(**outputs)` - Final result with all fields

```python
from udspy import OutputStreamChunk, Prediction

async for event in predictor.aexecute(stream=True, **inputs):
    match event:
        case OutputStreamChunk(field=field, delta=delta):
            print(f"[{field}] {delta}", end="")
        case Prediction() as pred:
            print(f"\n\nComplete: {pred}")
```

## Execution Methods

### Async Execution

```python
# With streaming
async for event in predictor.aexecute(stream=True, question="..."):
    ...

# Without streaming
result = await predictor.aforward(question="...")
```

### Synchronous Execution

```python
# Synchronous wrapper
result = predictor(question="...")
```

Internally, the synchronous call runs async code using `asyncio.run()` or the current event loop.

## History Management

Predict automatically manages conversation history:

```python
predictor = Predict(QA)

# First call
result1 = predictor(question="What is Python?")

# Second call - includes history
result2 = predictor(question="What are its key features?")

# Access history
for msg in predictor.history:
    print(f"{msg.role}: {msg.content}")

# Clear history
predictor.history.clear()
```

See [History API](../../api/history.md) for more details.

## Under the Hood

### Message Flow

1. **Signature → Messages**: Adapter converts signature to system prompt
2. **Inputs → User Message**: Input fields become user message
3. **LLM Call**: OpenAI API generates response
4. **Response → Outputs**: Parse response into output fields
5. **History Update**: Add messages to history

### Field Parsing

Outputs are parsed from the LLM response:

- **JSON Mode**: If response is valid JSON, parse it
- **Field Markers**: Look for field boundaries like `answer: ...`
- **Fallback**: Extract text content

The `Prediction` object makes all output fields accessible as attributes:

```python
result = predictor(question="...")
print(result.answer)  # Access via attribute
print(result["answer"])  # Access via dict key
```

## Error Handling and Retry

### Automatic Retry on Parse Errors

`Predict` automatically retries when LLM responses fail to parse correctly. This handles common issues like:
- Missing or malformed field markers
- Invalid JSON in structured outputs
- Format inconsistencies

**Retry behavior**:
- **Max attempts**: 3 (1 initial + 2 retries)
- **Backoff strategy**: Exponential backoff (0.1s, 0.2s, up to 3s)
- **Only retries**: `AdapterParseError` (not network errors or other exceptions)
- **Applies to**: Both streaming and non-streaming execution

```python
# This will automatically retry if parse fails
try:
    result = predictor(question="...")
except tenacity.RetryError as e:
    # All 3 attempts failed
    print(f"Failed after retries: {e}")
```

**Why automatic retry?**
- LLM format errors are usually transient and succeed on retry
- Reduces boilerplate error handling code
- Improves reliability without user intervention

See [ADR-007: Automatic Retry on Parse Errors](../decisions.md#adr-007-automatic-retry-on-parse-errors) for full details.

### Error Types

**AdapterParseError**: LLM response doesn't match expected format
- Automatically retried up to 3 times
- Check signature output fields match what LLM is generating

**ValidationError**: Output doesn't match Pydantic field types
- Not retried (indicates signature mismatch)
- Adjust signature types or field descriptions

**OpenAI API errors**: Network issues, rate limits, etc.
- Not retried automatically (use your own retry logic for these)
- Consider using exponential backoff for rate limits

## Design Rationale

### Why Not Auto-execute Tools by Default?

Tools are not auto-executed by default because:

1. **Safety**: User should control when external functions run
2. **Flexibility**: Sometimes you want to inspect/modify tool calls
3. **Explicit is better**: Makes behavior clear and predictable

### Why Native Tool Calling?

Unlike DSPy which uses custom adapters, udspy uses OpenAI's native function calling:

1. **Simplicity**: Less code to maintain
2. **Performance**: Optimized by OpenAI
3. **Reliability**: Well-tested and production-ready
4. **Features**: Access to latest tool calling improvements

See [ADR-001: Native Tool Calling](../decisions.md#adr-001-initial-project-setup) for full rationale.

## Common Patterns

### Simple Q&A

```python
qa = Predict("question -> answer")
result = qa(question="What is AI?")
```

### Multi-field Output

```python
analyzer = Predict("text -> summary, sentiment, keywords")
result = analyzer(text="I love this product!")
```

### With Context

```python
contextual = Predict("context, question -> answer")
result = contextual(
    context="Python is a programming language",
    question="What is it used for?"
)
```

### With Tools

```python
@tool(name="calculator")
def calc(expression: str = Field(...)) -> str:
    return str(eval(expression))

math_solver = Predict("problem -> solution", tools=[calc])
result = await math_solver.aexecute(
    problem="What is 157 * 234?",
    auto_execute_tools=True
)
```

## See Also

- [Base Module](base.md) - Module foundation
- [ChainOfThought](chain_of_thought.md) - Add reasoning
- [ReAct](react.md) - Agent with tool usage
- [Signatures](../signatures.md) - Define inputs/outputs
- [Tool Calling](../../api/tool.md) - Function calling API
- [Streaming](../streaming.md) - Real-time output
