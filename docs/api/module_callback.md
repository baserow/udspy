# Module Callbacks API

API reference for module callbacks - the mechanism that enables dynamic tool management during execution.

## Overview

Module callbacks are special callables that tools can return to modify the module during execution. They receive context about the current execution state and can add/remove tools, modify configuration, or perform other operations on the module.

---

## `@module_callback` Decorator

```python
@module_callback
def callback(context: ModuleContext) -> str:
    # Modify module
    context.module.init_module(tools=[...])
    return "Observation string"
```

Decorator that marks a function as a module callback. The decorated function:
- Receives a `ModuleContext` (or subclass) as its only parameter
- Must return a string that becomes the observation in the trajectory
- Can modify the module by calling `context.module.init_module()`

**Parameters:**

- **`func`** (`Callable[[ModuleContext], str]`): Function that takes context and returns observation string

**Returns:**

- `ModuleCallback`: Wrapped callback object

**Example:**

```python
from udspy import module_callback

@module_callback
def add_tools(context):
    # Access the module
    current_tools = list(context.module.tools.values())

    # Add new tools
    context.module.init_module(tools=current_tools + [new_tool])

    # Return observation
    return "Added new tool"
```

---

## `ModuleCallback` Class

```python
class ModuleCallback:
    """Wrapper for module callback functions."""

    func: Callable[[ModuleContext], str]

    def __call__(self, context: ModuleContext) -> str: ...
```

The `ModuleCallback` class wraps a callback function. You typically don't instantiate this directly; use the `@module_callback` decorator instead.

### Attributes

#### `func`

```python
func: Callable[[ModuleContext], str]
```

The underlying callback function.

### Methods

#### `__call__(context: ModuleContext) -> str`

Execute the callback with the provided context.

**Parameters:**

- **`context`** (`ModuleContext`): Execution context with module and state

**Returns:**

- `str`: Observation string to add to trajectory/history

---

## Context Classes

### `ModuleContext`

```python
class ModuleContext:
    """Base context for module callbacks."""

    module: Module
```

Base context class that provides access to the module instance.

**Attributes:**

- **`module`** (`Module`): The module instance (Predict or ReAct)

**Example:**

```python
@module_callback
def callback(context: ModuleContext):
    # Access module
    module = context.module

    # Access current tools
    tools = context.module.tools

    # Modify module
    context.module.init_module(tools=[...])

    return "Modified module"
```

---

### `PredictContext`

```python
class PredictContext(ModuleContext):
    """Context for Predict module callbacks."""

    module: Predict
    history: Optional[History]
```

Context for callbacks in Predict modules. Provides access to conversation history in addition to the module.

**Attributes:**

- **`module`** (`Predict`): The Predict module instance
- **`history`** (`Optional[History]`): Conversation history (if provided)

**Example:**

```python
from udspy.module.callbacks import PredictContext

@module_callback
def callback(context: PredictContext):
    # Access Predict-specific features
    module = context.module  # Type: Predict

    # Access conversation history
    if context.history:
        messages = context.history.messages
        print(f"Conversation has {len(messages)} messages")

    # Modify tools
    context.module.init_module(tools=[...])

    return "Updated tools based on conversation history"
```

---

### `ReactContext`

```python
class ReactContext(ModuleContext):
    """Context for ReAct module callbacks."""

    module: ReAct
    trajectory: dict[str, Any]
```

Context for callbacks in ReAct modules. Provides access to the agent's trajectory in addition to the module.

**Attributes:**

- **`module`** (`ReAct`): The ReAct module instance
- **`trajectory`** (`dict[str, Any]`): Current trajectory with thought/action/observation history

**Example:**

```python
from udspy.module.callbacks import ReactContext

@module_callback
def callback(context: ReactContext):
    # Access ReAct-specific features
    module = context.module  # Type: ReAct

    # Access trajectory
    trajectory = context.trajectory

    # Analyze agent's thoughts
    thoughts = [v for k, v in trajectory.items() if k.startswith("thought_")]
    print(f"Agent has had {len(thoughts)} thoughts")

    # Modify tools based on trajectory
    if len(thoughts) > 3:
        # Agent is struggling, add more tools
        context.module.init_module(tools=[...])

    return "Adapted tools based on trajectory"
```

**Trajectory Structure:**

The trajectory dictionary contains entries like:

```python
{
    "thought_0": "I need to search for information",
    "tool_calls_0": [{"name": "search", "args": {...}}],
    "observation_0": "Search results: ...",
    "thought_1": "Now I need to calculate",
    "tool_calls_1": [{"name": "calculator", "args": {...}}],
    "observation_1": "Result: 42",
    # ...
}
```

---

## Helper Functions

### `is_module_callback(obj: Any) -> bool`

```python
def is_module_callback(obj: Any) -> bool:
    """Check if an object is a module callback."""
```

Check if an object is a `ModuleCallback` instance.

**Parameters:**

- **`obj`** (`Any`): Object to check

**Returns:**

- `bool`: True if obj is a ModuleCallback instance

**Example:**

```python
from udspy import module_callback, is_module_callback

@module_callback
def my_callback(context):
    return "Done"

def regular_function():
    return "Done"

print(is_module_callback(my_callback))      # True
print(is_module_callback(regular_function))  # False
print(is_module_callback("not a function"))  # False
```

---

## Usage with Tools

Module callbacks are typically returned by tools to enable dynamic tool loading:

```python
from udspy import tool, module_callback
from pydantic import Field

@tool(name="load_calculator", description="Load calculator tool")
def load_calculator() -> callable:
    """Tool that returns a module callback."""

    @module_callback
    def add_calculator(context):
        # Get current tools
        current = [
            t for t in context.module.tools.values()
            if t.name not in ("finish", "ask_to_user")
        ]

        # Add calculator
        from my_tools import calculator
        context.module.init_module(tools=current + [calculator])

        return "Calculator loaded and ready to use"

    return add_calculator
```

**Execution Flow:**

1. LLM calls `load_calculator()` tool
2. Tool returns a `ModuleCallback` instance
3. Module detects the callback with `is_module_callback()`
4. Module creates appropriate context (`PredictContext` or `ReactContext`)
5. Module executes callback: `callback(context)`
6. Callback modifies module via `context.module.init_module()`
7. Callback returns observation string
8. Observation is added to history/trajectory
9. Execution continues with modified tool set

---

## Common Patterns

### Loading Tools on Demand

```python
@tool(name="load_tools", description="Load specialized tools")
def load_tools(category: str = Field(...)) -> callable:
    @module_callback
    def add_tools(context):
        current = list(context.module.tools.values())

        if category == "math":
            new_tools = [calculator, statistics]
        elif category == "web":
            new_tools = [search, scrape]

        context.module.init_module(tools=current + new_tools)
        return f"Loaded {len(new_tools)} {category} tools"

    return add_tools
```

### Conditional Loading Based on History

```python
@tool(name="smart_load", description="Intelligently load tools")
def smart_load() -> callable:
    @module_callback
    def analyze_and_load(context: PredictContext):
        # Analyze conversation history
        if context.history:
            messages = context.history.messages
            # Determine what tools are needed
            # ...

        context.module.init_module(tools=[...])
        return "Loaded appropriate tools"

    return analyze_and_load
```

### Progressive Tool Discovery

```python
@tool(name="discover_tools", description="Discover needed tools")
def discover_tools(task: str = Field(...)) -> callable:
    @module_callback
    def discover(context: ReactContext):
        # Analyze trajectory to see what agent has tried
        thoughts = [v for k, v in context.trajectory.items()
                   if k.startswith("thought_")]

        # Load tools the agent seems to need
        # ...

        context.module.init_module(tools=[...])
        return "Discovered and loaded needed tools"

    return discover
```

---

## Type Annotations

```python
from typing import Callable, Any
from udspy import Module, History

# Decorator
def module_callback(
    func: Callable[[ModuleContext], str]
) -> ModuleCallback: ...

# Callback class
class ModuleCallback:
    func: Callable[[ModuleContext], str]
    def __call__(self, context: ModuleContext) -> str: ...

# Context classes
class ModuleContext:
    module: Module

class PredictContext(ModuleContext):
    module: Predict
    history: Optional[History]

class ReactContext(ModuleContext):
    module: ReAct
    trajectory: dict[str, Any]

# Helper
def is_module_callback(obj: Any) -> bool: ...
```

---

## Important Notes

1. **Return String Required**: Callbacks MUST return a string - this becomes the observation

2. **Thread Safety**: Callbacks execute synchronously during tool execution

3. **Built-in Preservation**: ReAct's `finish` and `ask_to_user` tools are automatically preserved

4. **Tool Persistence**: Tools added via callbacks remain available for the entire execution

5. **Error Handling**: Wrap callback logic in try/except to handle errors gracefully:
   ```python
   @module_callback
   def safe_callback(context):
       try:
           context.module.init_module(tools=[...])
           return "Success"
       except Exception as e:
           return f"Failed: {e}"
   ```

---

## See Also

- [Dynamic Tools Guide](../examples/dynamic_tools.md) - Complete guide with examples
- [Tool API](tool.md) - Tool creation and usage
- [ReAct API](react.md) - ReAct module documentation
- [Predict API](module.md) - Predict module documentation

**Example Code**: See `examples/dynamic_calculator.py` and `examples/dynamic_tools.py` in the [GitHub repository](https://github.com/silvestrid/udspy/tree/main/examples)
