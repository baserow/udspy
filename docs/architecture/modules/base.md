# Base Module

The `Module` class is the foundation for all udspy modules. It provides a standard interface for composable LLM components.

## Purpose

The base module serves several key purposes:

1. **Unified Interface**: All modules implement the same execution methods (`aexecute`, `aforward`, `__call__`)
2. **Composition**: Modules can be nested and composed to build complex behaviors
3. **Streaming Support**: Built-in streaming infrastructure for real-time outputs
4. **Async-First**: Native async/await support for efficient I/O operations

## Core Methods

### `init_module(tools=None)`

Initialize or reinitialize the module with new tools. This is the key method for dynamic tool management.

**Why it's needed**: When tools are added or removed at runtime, the module needs to reconfigure its internal state. Specifically:

1. **Tool schemas must be regenerated** - The LLM needs updated JSON schemas for the new/different tools
2. **Signatures must be rebuilt** - Tool descriptions need to be incorporated into the prompt
3. **Tool dictionary must be updated** - The module needs to know which tools are available for execution

Without `init_module()`, adding a tool dynamically would be incomplete - the module would have the tool function but wouldn't know how to describe it to the LLM or include it in requests.

**Purpose**: Allows modules to rebuild their complete state (tools, schemas, signatures) during execution. This enables dynamic tool loading where tools can modify the available toolset.

**When to use**:
- Called from module callbacks (decorated with `@module_callback`)
- When you need to add/remove tools during execution
- When building adaptive agents that discover needed tools progressively

**Implementation requirements**:
1. Rebuild the tools dictionary
2. Regenerate tool schemas (if applicable)
3. Rebuild signatures with new tool descriptions (if applicable)
4. Preserve built-in tools (like ReAct's `finish` and user clarification)

```python
from udspy import module_callback

@module_callback
def add_tools(context):
    # Get current tools (excluding built-ins)
    current = [
        t for t in context.module.tools.values()
        if t.name not in ("finish", "user_clarification")
    ]

    # Add new tools
    new_tools = [calculator, weather_api]

    # Reinitialize module
    context.module.init_module(tools=current + new_tools)

    return "Added calculator and weather tools"
```

**See also**: [Dynamic Tool Management](../../examples/dynamic_tools.md) for detailed examples.

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
    if isinstance(event, OutputStreamChunk):
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
from udspy import Module, Predict, ChainOfThought, Prediction

class Pipeline(Module):
    def __init__(self):
        self.analyzer = Predict("text -> analysis")
        self.summarizer = ChainOfThought("text, analysis -> summary")

    async def aexecute(self, *, stream: bool = False, **inputs):
        # First module: get analysis (stream=False since we need full result)
        analysis = None
        async for event in self.analyzer.aexecute(stream=False, text=inputs["text"]):
            if isinstance(event, Prediction):
                analysis = event

        if not analysis:
            raise ValueError("First module did not produce a result")

        # Second module: pass down stream parameter
        async for event in self.summarizer.aexecute(
            stream=stream,
            text=inputs["text"],
            analysis=analysis.analysis
        ):
            # Yield all events from second module
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
