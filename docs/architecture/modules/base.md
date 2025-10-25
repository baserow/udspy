# Base Module

The `Module` class is the foundation for all udspy modules. It provides a standard interface for composable LLM components.

## Purpose

The base module serves several key purposes:

1. **Unified Interface**: All modules implement the same execution methods (`aexecute`, `aforward`, `__call__`)
2. **Composition**: Modules can be nested and composed to build complex behaviors
3. **Streaming Support**: Built-in streaming infrastructure for real-time outputs
4. **Async-First**: Native async/await support for efficient I/O operations

## Core Methods

### `aexecute(*, stream: bool = False, **inputs)`

The core execution method that all modules must implement. This is the public API for module execution.

- **stream**: If `True`, enables streaming mode for real-time output
- **inputs**: Keyword arguments matching the module's signature input fields
- **Returns**: `AsyncGenerator[StreamEvent, None]` that yields events and ends with a `Prediction`

```python
class CustomModule(Module):
    async def aexecute(self, *, stream: bool = False, **inputs):
        # Implementation here
        ...
        yield Prediction(result=final_result)
```

### `aforward(**inputs)`

Convenience method that calls `aexecute(stream=False)` and returns just the final `Prediction`.

```python
result = await module.aforward(question="What is Python?")
print(result.answer)
```

### `__call__(**inputs)`

Synchronous wrapper that runs `aforward` and returns the result. This is the most convenient way to use modules in synchronous code.

```python
result = module(question="What is Python?")
print(result.answer)
```

## Streaming Architecture

Modules support streaming through an async generator pattern:

```python
async for event in module.aexecute(stream=True, question="Explain AI"):
    if isinstance(event, StreamChunk):
        print(event.delta, end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\nFinal: {event.answer}")
```

The streaming system yields:
- `StreamChunk` events during generation (with `field` and `delta`)
- A final `Prediction` object with complete results

## Module Composition

Modules can contain other modules, creating powerful compositions:

```python
from udspy import Module, Predict, ChainOfThought

class Pipeline(Module):
    def __init__(self):
        self.analyzer = Predict("text -> analysis")
        self.summarizer = ChainOfThought("text, analysis -> summary")

    async def aexecute(self, *, stream: bool = False, **inputs):
        # First module
        analysis = await self.analyzer.aforward(text=inputs["text"])

        # Second module uses first module's output
        result = await self.summarizer.aexecute(
            stream=stream,
            text=inputs["text"],
            analysis=analysis.analysis
        )

        # Yield the final result
        async for event in result:
            yield event
```

## Design Rationale

### Why `aexecute` instead of `_aexecute`?

The method is named `aexecute` (public) rather than `_aexecute` (private) because:

1. **It's the public API**: Modules are meant to be executed via this method
2. **Subclasses override it**: Marking it private would be confusing since it's meant to be overridden
3. **Consistency**: Follows Python conventions where overridable methods are public

See [ADR-006](../decisions.md#adr-006-unified-module-execution-pattern-aexecute) for detailed rationale.

### Async-First Design

All modules are async-first because:

1. **I/O Bound**: LLM calls are network I/O operations
2. **Concurrent Operations**: Multiple LLM calls can run in parallel
3. **Streaming**: Async generators are ideal for streaming responses
4. **Modern Python**: Async/await is the standard for I/O-bound operations

The synchronous `__call__` wrapper provides convenience but internally uses async operations.

## Built-in Modules

udspy provides several built-in modules:

- **[Predict](predict.md)**: Core module for LLM predictions
- **[ChainOfThought](chain_of_thought.md)**: Adds reasoning before outputs
- **[ReAct](react.md)**: Reasoning and acting with tools

## Creating Custom Modules

To create a custom module:

1. Subclass `Module`
2. Implement `aexecute()` method
3. Yield `StreamEvent` objects during execution
4. Yield final `Prediction` at the end

```python
from udspy import Module, Prediction

class CustomModule(Module):
    async def aexecute(self, *, stream: bool = False, **inputs):
        # Your logic here
        result = process_inputs(inputs)

        # Return final prediction
        yield Prediction(output=result)
```

## See Also

- [Predict Module](predict.md) - The core prediction module
- [ChainOfThought Module](chain_of_thought.md) - Step-by-step reasoning
- [ReAct Module](react.md) - Agent with tool usage
- [ADR-006: Unified Execution Pattern](../decisions.md#adr-006-unified-module-execution-pattern-aexecute)
