# Modules

Modules are composable units that encapsulate LLM calls. They provide a standard interface for building complex LLM-powered applications.

## Overview

All modules inherit from the base `Module` class and implement a unified execution pattern. This allows modules to be composed, nested, and combined to create sophisticated behaviors.

## Core Concepts

### Unified Interface

Every module implements:
- `aexecute(*, stream: bool, **inputs)` - Core async execution
- `aforward(**inputs)` - Async convenience (no streaming)
- `__call__(**inputs)` - Synchronous wrapper

### Composition

Modules can contain other modules:

```python
from udspy import Module, Predict, ChainOfThought, Prediction

class Pipeline(Module):
    def __init__(self):
        self.analyze = Predict("text -> analysis")
        self.summarize = ChainOfThought("text, analysis -> summary")

    async def aexecute(self, *, stream: bool = False, **inputs):
        # First module: get analysis (stream=False since we need full result)
        analysis = None
        async for event in self.analyze.aexecute(stream=False, text=inputs["text"]):
            if isinstance(event, Prediction):
                analysis = event

        if not analysis:
            raise ValueError("First module did not produce a result")

        # Second module: pass down stream parameter
        async for event in self.summarize.aexecute(
            stream=stream,
            text=inputs["text"],
            analysis=analysis.analysis
        ):
            yield event
```

### Streaming Support

All modules support streaming for real-time output:

```python
async for event in module.aexecute(stream=True, **inputs):
    if isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\nFinal: {event}")
```

## Built-in Modules

udspy provides three core modules:

### [Base Module](modules/base.md)

The foundation for all modules. Provides:
- Unified execution interface
- Streaming infrastructure
- Async-first design
- Composition support

**When to use**: When creating custom modules

### [Predict](modules/predict.md)

The core module for LLM predictions. Features:
- Maps signature inputs to outputs
- Native tool calling support
- Conversation history management
- Streaming and async execution

**When to use**: For basic LLM calls, tool usage, and as a building block for other modules

```python
predictor = Predict("question -> answer")
result = predictor(question="What is AI?")
```

### [ChainOfThought](modules/chain_of_thought.md)

Adds step-by-step reasoning before outputs. Features:
- Automatic reasoning field injection
- Improves answer quality on complex tasks
- Transparent reasoning process
- Works with any signature

**When to use**: For tasks requiring reasoning (math, analysis, decision-making)

```python
cot = ChainOfThought("question -> answer")
result = cot(question="What is 157 * 234?")
print(result.reasoning)  # Shows step-by-step work
print(result.answer)     # "36738"
```

### [ReAct](modules/react.md)

Agent that reasons and acts with tools. Features:
- Iterative reasoning and tool usage
- Human-in-the-loop support
- Built-in ask_to_user and finish tools
- Full trajectory tracking

**When to use**: For tasks requiring multiple steps, tool usage, or agent-like behavior

```python
@tool(name="search")
def search(query: str = Field(...)) -> str:
    return search_web(query)

agent = ReAct("question -> answer", tools=[search])
result = agent(question="What's the weather in Tokyo?")
```

## Module Comparison

| Feature | Predict | ChainOfThought | ReAct |
|---------|---------|----------------|-------|
| Basic LLM calls | ✅ | ✅ | ✅ |
| Step-by-step reasoning | ❌ | ✅ | ✅ |
| Tool usage | ✅ | ✅ | ✅ |
| Multi-step iteration | ❌ | ❌ | ✅ |
| Human-in-the-loop | ❌ | ❌ | ✅ |
| Trajectory tracking | ❌ | ❌ | ✅ |
| Complexity | Low | Low | Medium |
| Token usage | Low | Medium | High |
| Latency | Low | Medium | High |

## Creating Custom Modules

To create a custom module:

1. Subclass `Module`
2. Implement `aexecute()` method
3. Yield `StreamEvent` objects during execution
4. Yield final `Prediction` at the end

```python
from udspy import Module, Prediction, OutputStreamChunk

class CustomModule(Module):
    def __init__(self, signature):
        self.predictor = Predict(signature)

    async def aexecute(self, *, stream: bool = False, **inputs):
        # Custom logic before prediction
        processed_inputs = preprocess(inputs)

        # Call nested module's aexecute, passing down stream parameter
        async for event in self.predictor.aexecute(stream=stream, **processed_inputs):
            if isinstance(event, Prediction):
                # Custom logic after prediction
                final_result = postprocess(event)
                # Yield final prediction
                yield Prediction(**final_result)
            else:
                # Pass through other events (OutputStreamChunks, etc.)
                yield event
```

See [Base Module](modules/base.md) for detailed guidance.

## Best Practices

### Choose the Right Module

- **Simple tasks**: Use `Predict`
- **Need reasoning**: Use `ChainOfThought`
- **Need tools/agents**: Use `ReAct`
- **Custom logic**: Create custom module

### Composition over Inheritance

Build complex behaviors by composing modules rather than deep inheritance:

```python
# Good: Composition
class Pipeline(Module):
    def __init__(self):
        self.step1 = Predict(sig1)
        self.step2 = ChainOfThought(sig2)

# Avoid: Deep inheritance
class MyComplexModule(ChainOfThought):
    # Complex overrides
```

### Async Best Practices

- Use `aforward()` when you don't need streaming
- Use `aexecute(stream=True)` for real-time output
- Use `__call__()` only in sync contexts
- Always `await` async operations

### Error Handling

```python
try:
    result = await module.aforward(**inputs)
except ConfirmationRequired as e:
    # Handle confirmations
    result = await module.aresume(user_input, e)
except Exception as e:
    # Handle other errors
    logger.error(f"Module failed: {e}")
```

## See Also

- [Base Module](modules/base.md) - Module foundation
- [Predict Module](modules/predict.md) - Core prediction
- [ChainOfThought Module](modules/chain_of_thought.md) - Step-by-step reasoning
- [ReAct Module](modules/react.md) - Agent with tools
- [Signatures](signatures.md) - Define inputs/outputs
- [Streaming](streaming.md) - Real-time output
- [API: Modules](../api/module.md) - Full API reference
