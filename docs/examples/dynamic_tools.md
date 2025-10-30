# Dynamic Tool Management

Dynamic tool management allows tools to modify the module during execution by adding or removing other tools. This is useful when you need specialized tools that should only be loaded on demand.

## Overview

Tools can return **module callbacks** - special callables decorated with `@module_callback` that receive execution context and can call `init_module()` to modify available tools.

### Key Concepts

1. **Module Callbacks**: Functions decorated with `@module_callback` that can modify module state
2. **Tool Loaders**: Tools that return module callbacks to add other tools
3. **Context Objects**: Provide access to the module instance and execution state
4. **Dynamic Loading**: Tools are added during execution and persist until completion

## Basic Example: Calculator

Here's a complete example showing a ReAct agent that dynamically loads a calculator tool:

```python
from pydantic import Field
import udspy
from udspy import ReAct, Signature, tool, module_callback, InputField, OutputField

# Define the calculator tool (not initially available)
@tool(name="calculator", description="Perform mathematical calculations")
def calculator(expression: str = Field(...)) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Define the tool loader
@tool(name="load_calculator", description="Load the calculator tool")
def load_calculator() -> callable:
    """Load calculator tool dynamically."""

    @module_callback
    def add_calculator(context):
        # Get current tools (excluding built-ins)
        current_tools = [
            t for t in context.module.tools.values()
            if t.name not in ("finish", "ask_to_user")
        ]

        # Add calculator to available tools
        context.module.init_module(tools=current_tools + [calculator])

        return "Calculator loaded successfully"

    return add_calculator

# Create agent with only the loader
class Question(Signature):
    """Answer questions. Load tools if needed."""
    question: str = InputField()
    answer: str = OutputField()

agent = ReAct(Question, tools=[load_calculator])

# Agent will load calculator when needed
result = agent(question="What is 157 * 834?")
print(result.answer)  # 130938
```

### Trajectory Breakdown

When the agent processes "What is 157 * 834?", the trajectory looks like:

1. **Thought**: "I need to calculate 157 * 834. I don't have a calculator tool, so I'll load it first."
2. **Tool Call**: `load_calculator()` → Returns module callback
3. **Callback Execution**: Calculator tool is added to available tools
4. **Observation**: "Calculator loaded successfully"
5. **Thought**: "Now I can use the calculator"
6. **Tool Call**: `calculator(expression="157 * 834")` → Returns "Result: 130938"
7. **Observation**: "Result: 130938"
8. **Thought**: "I have the answer"
9. **Tool Call**: `finish()` with answer "130938"

## Context Objects

Module callbacks receive a context object with access to the module and execution state:

### ReactContext

For ReAct modules, the context includes the trajectory:

```python
@module_callback
def my_callback(context: ReactContext):
    # Access the module
    module = context.module

    # Access current tools
    current_tools = list(context.module.tools.values())

    # Access trajectory history
    trajectory = context.trajectory
    thoughts = [v for k, v in trajectory.items() if k.startswith("thought_")]

    # Modify tools
    context.module.init_module(tools=current_tools + [new_tool])

    return "Tools updated"
```

### PredictContext

For Predict modules, the context includes conversation history:

```python
@module_callback
def my_callback(context: PredictContext):
    # Access the module
    module = context.module

    # Access conversation history
    history = context.history
    if history:
        messages = history.messages

    # Modify tools
    context.module.init_module(tools=[...])

    return "Tools updated"
```

## Advanced Patterns

### Category-Based Loading

Load different tool sets based on categories:

```python
@tool(name="load_tools", description="Load specialized tools")
def load_tools(category: str = Field(...)) -> callable:
    """Load tools for a specific category."""

    @module_callback
    def add_tools(context):
        current = [
            t for t in context.module.tools.values()
            if t.name not in ("finish", "ask_to_user", "load_tools")
        ]

        new_tools = []
        if category == "math":
            new_tools = [calculator, statistics_tool]
        elif category == "web":
            new_tools = [search_tool, scrape_tool]
        elif category == "data":
            new_tools = [csv_tool, json_tool]

        context.module.init_module(tools=current + new_tools)

        tool_names = [t.name for t in new_tools]
        return f"Loaded {len(new_tools)} tools: {', '.join(tool_names)}"

    return add_tools
```

### Conditional Tool Loading

Load tools based on context analysis:

```python
@tool(name="analyze_and_load", description="Analyze question and load needed tools")
def analyze_and_load(question: str = Field(...)) -> callable:
    """Analyze question and load appropriate tools."""

    @module_callback
    def smart_load(context):
        current = list(context.module.tools.values())
        new_tools = []

        # Analyze what's needed
        if any(word in question.lower() for word in ["calculate", "multiply", "add"]):
            new_tools.append(calculator)
        if any(word in question.lower() for word in ["weather", "temperature"]):
            new_tools.append(weather_tool)
        if any(word in question.lower() for word in ["search", "find", "lookup"]):
            new_tools.append(search_tool)

        context.module.init_module(tools=current + new_tools)

        return f"Loaded {len(new_tools)} tools based on question analysis"

    return smart_load
```

### Tool Replacement

Replace existing tools with updated versions:

```python
@tool(name="upgrade_tools", description="Upgrade to advanced versions")
def upgrade_tools() -> callable:
    """Replace basic tools with advanced versions."""

    @module_callback
    def do_upgrade(context):
        # Remove basic tools
        current = [
            t for t in context.module.tools.values()
            if not t.name.startswith("basic_")
        ]

        # Add advanced tools
        advanced = [advanced_calculator, advanced_search]

        context.module.init_module(tools=current + advanced)

        return "Upgraded to advanced tools"

    return do_upgrade
```

## Important Notes

### Tool Persistence

Tools loaded via module callbacks **persist for the entire execution**:

```python
agent = ReAct(Question, tools=[load_calculator])

# First call - loads calculator
result1 = agent(question="What is 10 * 5?")
# Calculator is now in agent.tools

# Second call - calculator still available from before
result2 = agent(question="What is 20 + 15?")
```

If you want fresh tool state, create a new agent instance.

### Return Values

Module callbacks **must return a string** that becomes the observation in the trajectory:

```python
@module_callback
def callback(context):
    context.module.init_module(tools=[...])
    return "This string appears in the trajectory"  # Required!
```

### Built-in Tools

ReAct's built-in tools (`finish`, `ask_to_user`) are automatically preserved when you call `init_module()`. You don't need to include them manually.

### Error Handling

Handle errors gracefully in your callbacks:

```python
@module_callback
def safe_callback(context):
    try:
        # Load tools
        context.module.init_module(tools=[...])
        return "Success"
    except Exception as e:
        return f"Failed to load tools: {e}"
```

## Use Cases

### 1. On-Demand Capabilities

Load expensive or specialized tools only when needed:

```python
# Start lightweight, load heavy tools on demand
agent = ReAct(Task, tools=[load_nlp_tools, load_vision_tools])
```

### 2. Progressive Tool Discovery

Agent discovers what tools it needs as it works:

```python
# Agent figures out it needs web search, then calculator, then data analysis
agent = ReAct(Task, tools=[load_tools])
```

### 3. Multi-Tenant Applications

Load user-specific or permission-based tools:

```python
@tool(name="load_user_tools")
def load_user_tools(user_id: str) -> callable:
    @module_callback
    def add_user_tools(context):
        user_tools = get_tools_for_user(user_id)
        context.module.init_module(tools=user_tools)
        return f"Loaded tools for user {user_id}"
    return add_user_tools
```

### 4. Adaptive Tool Sets

Adjust tools based on task complexity:

```python
@tool(name="assess_and_load")
def assess_and_load(task: str) -> callable:
    @module_callback
    def adaptive_load(context):
        if is_complex(task):
            tools = advanced_tools
        else:
            tools = basic_tools
        context.module.init_module(tools=tools)
        return "Tools loaded based on task complexity"
    return adaptive_load
```

## See Also

- [ReAct API Reference](../api/react.md)
- [Tool Calling Guide](tool_calling.md)
- [Module Callbacks API](../api/callback.md)
- [Example: dynamic_calculator.py](../../examples/dynamic_calculator.py)
- [Example: dynamic_tools.py](../../examples/dynamic_tools.py)
