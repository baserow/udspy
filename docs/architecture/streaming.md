# Streaming

Streaming support allows incremental processing of LLM outputs, providing real-time feedback to users.

## Overview

All modules support streaming out of the box through the `astream()` method:

```python
from udspy import Predict, OutputStreamChunk, Prediction

predictor = Predict("question -> answer")

async for event in predictor.astream(question="Explain AI"):
    if isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\n\nFinal result: {event.answer}")
```

## Stream Events

The streaming API yields two types of events:

### OutputStreamChunk

Incremental text updates for a specific field:

```python
class OutputStreamChunk(StreamEvent):
    module: Module          # Module that generated this chunk
    field_name: str        # Which output field (e.g., "answer")
    delta: str             # New text since last chunk
    content: str           # Full accumulated text so far
    is_complete: bool      # Whether field is done streaming
```

Example:
```python
async for event in predictor.astream(question="..."):
    if isinstance(event, OutputStreamChunk):
        print(f"[{event.field_name}] {event.delta}", end="", flush=True)
        if event.is_complete:
            print(f"\n--- {event.field_name} complete ---")
```

### Prediction

Final result with all output fields:

```python
class Prediction(StreamEvent, dict):
    # Dict with all output fields
    # Supports both dict and attribute access
```

Example:
```python
async for event in predictor.astream(question="..."):
    if isinstance(event, Prediction):
        print(f"Answer: {event.answer}")
        print(f"Same: {event['answer']}")
```

## Field-Specific Streaming

Streaming automatically handles multiple output fields:

```python
from udspy import ChainOfThought

cot = ChainOfThought("question -> answer")

async for event in cot.astream(question="What is 157 * 234?"):
    if isinstance(event, OutputStreamChunk):
        if event.field_name == "reasoning":
            print(f"💭 {event.delta}", end="", flush=True)
        elif event.field_name == "answer":
            print(f"\n✓ {event.delta}", end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\n\nComplete!")
```

## Custom Stream Events

You can emit custom events from tools or callbacks:

```python
from dataclasses import dataclass
from udspy.streaming import StreamEvent, emit_event

@dataclass
class ToolProgress(StreamEvent):
    tool_name: str
    message: str
    progress: float  # 0.0 to 1.0

# In your tool:
from udspy import tool
from pydantic import Field

@tool(name="search")
async def search(query: str = Field(...)) -> str:
    await emit_event(ToolProgress("search", "Starting search...", 0.0))

    results = await search_api(query)

    await emit_event(ToolProgress("search", "Processing results...", 0.5))

    processed = process_results(results)

    await emit_event(ToolProgress("search", "Complete!", 1.0))

    return processed

# In the stream consumer:
async for event in predictor.astream(question="..."):
    if isinstance(event, ToolProgress):
        print(f"[{event.tool_name}] {event.message} ({event.progress*100:.0f}%)")
    elif isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)
```

## Module Support

All built-in modules support streaming:

### Predict

```python
predictor = Predict("question -> answer")
async for event in predictor.astream(question="..."):
    ...
```

### ChainOfThought

Streams both reasoning and answer:

```python
cot = ChainOfThought("question -> answer")
async for event in cot.astream(question="..."):
    if isinstance(event, OutputStreamChunk):
        if event.field_name == "reasoning":
            # Reasoning streams first
            ...
        elif event.field_name == "answer":
            # Answer streams second
            ...
```

### ReAct

Streams reasoning and tool interactions:

```python
from udspy import ReAct

agent = ReAct("question -> answer", tools=[search])
async for event in agent.astream(question="..."):
    if isinstance(event, OutputStreamChunk):
        if event.field_name == "reasoning":
            print(f"💭 {event.delta}", end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\n\n✓ {event.answer}")
```

See `examples/react_streaming.py` for a complete example.

## Implementation Details

### Context Variables

Streaming uses Python's `contextvars` for thread-safe event queuing:

```python
from udspy.streaming import _stream_queue, emit_event

# Internal: stream queue is set when aexecute(stream=True) is called
# emit_event() checks if a queue exists and puts events there
```

This allows tools and nested modules to emit events without explicit queue passing.

### Event Flow

1. Module's `astream(**inputs)` is called
2. Queue is created and set in context
3. Internal `aexecute(stream=True)` is called
4. Module yields `StreamChunk` events as text arrives
5. Tools can call `emit_event()` to inject custom events
6. Module yields final `Prediction` when complete
7. Queue is cleaned up

### Non-Streaming Mode

For non-streaming execution, use `aforward()` instead of `astream()`:

```python
# Streaming: iterate over events
async for event in predictor.astream(question="..."):
    if isinstance(event, Prediction):
        result = event

# Non-streaming: get final result directly
result = await predictor.aforward(question="...")
```

## Best Practices

### 1. Always Handle Both Event Types

```python
async for event in module.astream(**inputs):
    match event:
        case OutputStreamChunk():
            # Handle streaming text
            print(event.delta, end="", flush=True)
        case Prediction():
            # Handle final result
            final_result = event
```

### 2. Check Field Names for Multi-field Outputs

```python
async for event in module.astream(**inputs):
    if isinstance(event, OutputStreamChunk):
        if event.field_name == "reasoning":
            # Different formatting for reasoning
            print(f"💭 {event.delta}", end="")
        elif event.field_name == "answer":
            # Different formatting for answer
            print(f"✓ {event.delta}", end="")
```

### 3. Use Custom Events for Progress

```python
@dataclass
class Progress(StreamEvent):
    step: str
    percent: float

async def long_running_tool():
    await emit_event(Progress("Loading data", 0.3))
    data = load_data()

    await emit_event(Progress("Processing", 0.6))
    result = process(data)

    await emit_event(Progress("Complete", 1.0))
    return result
```

### 4. Accumulate Chunks for Display

```python
accumulated = {}

async for event in module.astream(**inputs):
    if isinstance(event, OutputStreamChunk):
        field = event.field_name
        if field not in accumulated:
            accumulated[field] = ""
        accumulated[field] += event.delta

        # Update UI with accumulated[field]
        update_display(field, accumulated[field])
```

## Performance Considerations

### Latency

Streaming reduces perceived latency by showing results immediately:

- **Non-streaming**: Wait for full response (~5s), then show all text
- **Streaming**: Start showing text after ~500ms, continue as it arrives

### Token Usage

Streaming doesn't affect token usage - same number of tokens are generated.

### Error Handling

Errors can occur mid-stream:

```python
try:
    async for event in module.astream(**inputs):
        if isinstance(event, OutputStreamChunk):
            print(event.delta, end="", flush=True)
except Exception as e:
    print(f"\n\nError during streaming: {e}")
```

## See Also

- [API: Streaming](../api/streaming.md) - Full API reference
- [Examples: Streaming](../examples/streaming.md) - Complete examples
- [Examples: ReAct Streaming](../examples/react.md) - Agent streaming
- [Base Module](modules/base.md) - Module execution patterns
