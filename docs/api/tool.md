# Tool API Reference

API documentation for creating and using tools with native OpenAI function calling.

## Module: `udspy.tool`

### `@tool` Decorator

```python
@tool(
    name: str | None = None,
    description: str | None = None,
    *,
    require_confirmation: bool = False,
) -> Callable[[Callable], Tool]
```

Decorator to mark a function as a tool for use with `Predict` and `ReAct` modules.

**Parameters:**

- **`name`** (`str | None`, default: `None`): Tool name visible to the LLM
  - If not provided, uses the function name
- **`description`** (`str | None`, default: `None`): Tool description visible to the LLM
  - If not provided, uses the function's docstring
  - This helps the LLM decide when to use the tool
- **`require_confirmation`** (`bool`, default: `False`): Whether to require user confirmation before execution
  - If `True`, wraps the tool with `@confirm_first` decorator
  - Raises `ConfirmationRequired` on first call, executes after approval
  - Useful for destructive or sensitive operations

**Returns:**

- `Tool`: A wrapped tool object with metadata

**Example:**

```python
from pydantic import Field
from udspy import tool

@tool(name="calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, or divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Execute arithmetic operation."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]
```

### Tool Confirmation Example

```python
import os
from pydantic import Field
from udspy import tool

@tool(
    name="delete_file",
    description="Delete a file from the filesystem",
    require_confirmation=True  # Requires user confirmation
)
def delete_file(path: str = Field(description="File path to delete")) -> str:
    """Delete a file (requires confirmation)."""
    os.remove(path)
    return f"Deleted {path}"
```

---

## `Tool` Class

```python
class Tool:
    """Wrapper for a tool function with metadata."""
```

The `Tool` class wraps a function and adds metadata for OpenAI function calling. You typically don't instantiate this directly; use the `@tool` decorator instead.

### Constructor

```python
def __init__(
    self,
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    *,
    require_confirmation: bool = False,
    desc: str | None = None,
    args: dict[str, str] | None = None,
)
```

**Parameters:**

- **`func`** (`Callable`): The function to wrap
- **`name`** (`str | None`): Tool name (defaults to function name)
- **`description`** (`str | None`): Tool description (defaults to docstring)
- **`require_confirmation`** (`bool`, default: `False`): Whether to require confirmation before execution
- **`desc`** (`str | None`): Alias for `description` (DSPy compatibility)
- **`args`** (`dict[str, str] | None`): Manual argument specification (DSPy compatibility)

**Example:**

```python
from udspy import Tool

def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

tool = Tool(my_function, name="adder", description="Adds numbers")
```

However, using the `@tool` decorator is preferred:

```python
@tool(name="adder", description="Adds numbers")
def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

### Attributes

#### `func`

```python
func: Callable
```

The underlying function that this tool wraps.

#### `name`

```python
name: str
```

The tool's name as seen by the LLM.

#### `description`

```python
description: str
```

The tool's description as seen by the LLM.

#### `require_confirmation`

```python
require_confirmation: bool
```

Whether this tool requires user confirmation before execution.

#### `parameters`

```python
parameters: dict[str, dict[str, Any]]
```

Dictionary mapping parameter names to their metadata:

```python
{
    "param_name": {
        "type": str,  # Python type
        "description": "Parameter description",
        "required": True  # Whether parameter is required
    },
    # ...
}
```

#### `desc` (DSPy compatibility)

```python
desc: str
```

Alias for `description`. Provided for DSPy compatibility.

#### `args` (DSPy compatibility)

```python
args: dict[str, str]
```

Dictionary mapping parameter names to type + description strings. Provided for DSPy compatibility.

**Example:**

```python
{
    "operation": "str - add, subtract, multiply, or divide",
    "a": "float - First number",
    "b": "float - Second number"
}
```

---

### Methods

#### `__call__(*args, **kwargs) -> Any`

Call the wrapped function synchronously.

**Parameters:**

- **`*args`**: Positional arguments
- **`**kwargs`**: Keyword arguments

**Returns:**

- Function result

**Example:**

```python
@tool(name="add", description="Add numbers")
def add(a: int, b: int) -> int:
    return a + b

result = add(2, 3)  # Returns 5
```

#### `acall(**kwargs) -> Any`

Call the wrapped function asynchronously.

- If the function is async, awaits it directly
- If the function is sync, runs it in an executor to avoid blocking (unless `require_confirmation=True`)
- If `require_confirmation=True`, may raise `ConfirmationRequired` before execution

**Parameters:**

- **`**kwargs`**: Keyword arguments

**Returns:**

- Awaitable that resolves to the function result

**Raises:**

- `ConfirmationRequired`: If `require_confirmation=True` and not yet approved

**Example:**

```python
import asyncio

@tool(name="fetch", description="Fetch data")
async def fetch_data(url: str) -> str:
    # Async operation
    return f"Data from {url}"

# Call async
result = await fetch_data.acall(url="https://example.com")
```

**Sync function example:**

```python
@tool(name="compute", description="Compute value")
def compute(x: int) -> int:
    return x * 2

# Still works with acall - runs in executor
result = await compute.acall(x=5)
```

#### `parameters` Property

Get the JSON schema for OpenAI function calling.

**Returns:**

- `dict`: Complete JSON schema with type, properties, and required fields

**Example:**

```python
@tool(name="calculator", description="Do math")
def calculator(
    operation: str = Field(description="Operation type"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    return eval(f"{a} {operation} {b}")

# Get the parameters schema for OpenAI
params = calculator.parameters
# {
#     "type": "object",
#     "properties": {
#         "operation": {
#             "type": "string",
#             "description": "Operation type"
#         },
#         "a": {
#             "type": "number",
#             "description": "First number"
#         },
#         "b": {
#             "type": "number",
#             "description": "Second number"
#         }
#     },
#     "required": ["operation", "a", "b"]
# }

# The adapter uses this to build OpenAI function schemas:
from udspy.adapter import ChatAdapter
adapter = ChatAdapter()
openai_schema = adapter.format_tool_schema(calculator)
# {
#     "type": "function",
#     "function": {
#         "name": "calculator",
#         "description": "Do math",
#         "parameters": calculator.parameters  # ← Uses this property
#     }
# }
```

#### `format() -> str`

Format the tool as a human-readable string for LLM prompts.

**Returns:**

- `str`: Human-readable description with name, description, and parameters

**Example:**

```python
@tool(name="calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="Operation type"),
    a: float = Field(description="First number"),
) -> float:
    return a * 2

# Get human-readable format for module prompts
description = calculator.format()
# "calculator, whose description is <desc>Perform arithmetic operations</desc>. It takes arguments {<properties>}."
```

---

## Parameter Type Annotations

Tools use Python type hints to generate OpenAI schemas. Supported types:

| Python Type | JSON Schema Type |
|-------------|------------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| `Optional[T]` | Type of `T` (nullable) |

**Example:**

```python
from typing import Optional
from pydantic import Field
from udspy import tool

@tool(name="search", description="Search with filters")
def search(
    query: str = Field(description="Search query"),
    max_results: int = Field(description="Maximum results", default=10),
    include_archived: bool = Field(description="Include archived", default=False),
    tags: Optional[list] = Field(description="Filter by tags", default=None),
) -> str:
    return f"Searching for: {query}"
```

---

## Using Pydantic Fields

Use `pydantic.Field()` to add parameter descriptions and defaults:

```python
from pydantic import Field

@tool(name="example", description="Example tool")
def example_tool(
    # Required parameter with description
    query: str = Field(description="The search query"),

    # Optional parameter with default
    limit: int = Field(description="Result limit", default=10),

    # Optional parameter that can be None
    filter: Optional[str] = Field(description="Optional filter", default=None),
) -> str:
    return f"Query: {query}, Limit: {limit}"
```

**Important:**

- Always provide descriptions for parameters
- Use `Field(...)` or `Field()` for required parameters (no default)
- Use `Field(default=value)` for optional parameters
- Descriptions help the LLM understand when and how to use the tool

---

## Tool Confirmation

For destructive or sensitive operations, use `require_confirmation=True`:

```python
import os
from pydantic import Field
from udspy import tool, ConfirmationRequired

@tool(
    name="delete_all_files",
    description="Delete all files in a directory",
    require_confirmation=True  # Requires confirmation
)
def delete_all_files(
    directory: str = Field(description="Directory path")
) -> str:
    """Delete all files in directory (requires confirmation)."""
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
    return f"Deleted all files in {directory}"

# When ReAct tries to call this tool, it raises ConfirmationRequired on first call
# After user approves, the tool executes normally
```

**How it works:**

1. LLM decides to call the tool
2. Tool function (wrapped with `@confirm_first`) raises `ConfirmationRequired` on first call
3. User sees confirmation prompt: `"Confirm execution of delete_all_files with args: {...}? (yes/no)"`
4. User responds with `"yes"`, `"no"`, or modified arguments
5. ReAct resumes execution based on user's response
6. If approved, subsequent calls to the same tool with same args execute normally

---

## Usage with ReAct

```python
from pydantic import Field
from udspy import InputField, OutputField, ReAct, Signature, tool

# Define tools
@tool(name="search", description="Search for information")
def search(query: str = Field(description="Search query")) -> str:
    return f"Results for: {query}"

@tool(name="calculate", description="Perform calculations")
def calculate(expression: str = Field(description="Math expression")) -> str:
    return str(eval(expression))

# Define signature
class QA(Signature):
    """Answer questions using tools."""
    question: str = InputField()
    answer: str = OutputField()

# Create agent with tools
agent = ReAct(QA, tools=[search, calculate])

# Execute
result = agent(question="What is the population of Tokyo times 2?")
# Agent will:
# 1. Call search("Tokyo population")
# 2. Call calculate("population * 2")
# 3. Synthesize final answer
```

---

## Usage with Predict

```python
from udspy import Predict, Signature, InputField, OutputField, tool
from pydantic import Field

@tool(name="get_weather", description="Get current weather")
def get_weather(city: str = Field(description="City name")) -> str:
    return f"Weather in {city}: Sunny, 72°F"

class WeatherQuery(Signature):
    """Get weather information."""
    city: str = InputField()
    weather: str = OutputField()

predictor = Predict(WeatherQuery, tools=[get_weather])
result = predictor(city="San Francisco")
```

---

## DSPy Compatibility

The `Tool` class includes DSPy-compatible attributes:

```python
@tool(name="search", description="Search tool")
def search(query: str = Field(description="Search query")) -> str:
    return "results"

# DSPy-style access
print(search.desc)  # Same as search.description: "Search tool"
print(search.args)  # Properties dict: {"query": {"type": "string", "description": "Search query", ...}}
```

---

## Best Practices

1. **Clear descriptions**: Write clear, concise tool and parameter descriptions
   ```python
   @tool(
       name="search_papers",
       description="Search academic papers by keyword, author, or topic"
   )
   ```

2. **Use Field() for all parameters**: Always use `Field()` with descriptions
   ```python
   def search(
       query: str = Field(description="Search query"),
       year: Optional[int] = Field(description="Publication year", default=None)
   )
   ```

3. **Require confirmation for destructive ops**: Use `require_confirmation=True`
   ```python
   @tool(name="delete", description="Delete data", require_confirmation=True)
   ```

4. **Handle errors gracefully**: Return error messages as strings
   ```python
   @tool(name="divide", description="Divide numbers")
   def divide(a: float = Field(...), b: float = Field(...)) -> str:
       if b == 0:
           return "Error: Cannot divide by zero"
       return str(a / b)
   ```

5. **Keep tools focused**: Each tool should do one thing well
   ```python
   # Good: Focused tool
   @tool(name="search_users", description="Search for users")
   def search_users(query: str = Field(...)) -> str: ...

   # Bad: Too many responsibilities
   @tool(name="user_management", description="Manage all user operations")
   def user_management(action: str, query: str, data: dict) -> str: ...
   ```

---

## Type Annotations

```python
from typing import Callable, Any
from udspy import Tool

# Decorator signature
def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    require_confirmation: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...

# Tool class
class Tool:
    func: Callable[..., Any]
    name: str
    description: str
    require_confirmation: bool
    parameters: dict[str, dict[str, Any]]
    desc: str  # Alias for description
    args: dict[str, str]  # DSPy compatibility

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    async def acall(self, **kwargs: Any) -> Any: ...
    def to_openai_schema(self) -> dict[str, Any]: ...
```

---

## Dynamic Tool Management

Tools can return **module callbacks** to dynamically add other tools during execution. This is useful for:

- Loading specialized tools on demand
- Adding tools based on user permissions
- Progressive tool discovery

### Basic Example

```python
from udspy import tool, module_callback
from pydantic import Field

# The tool that will be loaded dynamically
@tool(name="calculator", description="Perform calculations")
def calculator(expression: str = Field(...)) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

# Meta-tool that loads the calculator
@tool(name="load_calculator", description="Load calculator tool")
def load_calculator() -> callable:
    """Load calculator dynamically."""

    @module_callback
    def add_calculator(context):
        # Get current tools (excluding built-ins)
        current = [
            t for t in context.module.tools.values()
            if t.name not in ("finish", "user_clarification")
        ]

        # Add calculator
        context.module.init_module(tools=current + [calculator])

        return "Calculator loaded successfully"

    return add_calculator

# Use with ReAct
from udspy import ReAct, Signature, InputField, OutputField

class Question(Signature):
    """Answer questions. Load tools if needed."""
    question: str = InputField()
    answer: str = OutputField()

agent = ReAct(Question, tools=[load_calculator])
result = agent(question="What is 157 * 834?")
# Agent will:
# 1. Call load_calculator() to get the calculator tool
# 2. Use calculator(expression="157 * 834")
# 3. Return the answer
```

### How It Works

1. **Tool returns callable**: Instead of a string/value, return a function decorated with `@module_callback`
2. **Callback receives context**: Context has the module instance and execution state
3. **Callback modifies tools**: Call `context.module.init_module(tools=[...])` to add/remove tools
4. **Callback returns observation**: Return a string that appears in the trajectory
5. **Tools persist**: Newly added tools remain available for the rest of execution

### Category-Based Loading

```python
@tool(name="load_tools", description="Load tools by category")
def load_tools(category: str = Field(...)) -> callable:
    @module_callback
    def add_tools(context):
        current = list(context.module.tools.values())

        if category == "math":
            new_tools = [calculator, statistics]
        elif category == "web":
            new_tools = [search, scrape]
        else:
            return f"Unknown category: {category}"

        context.module.init_module(tools=current + new_tools)
        return f"Loaded {len(new_tools)} {category} tools"

    return add_tools
```

### Important Notes

- **Must return string**: Module callbacks must return a string (the observation)
- **Async by default**: Callbacks are called during tool execution
- **Built-ins preserved**: `finish` and user clarification are automatically kept
- **Persistence**: Tools remain available for the entire execution

For complete documentation and examples, see:
- [Dynamic Tools Guide](../examples/dynamic_tools.md) - Complete guide with examples

**Example Code**: See `examples/dynamic_calculator.py` and `examples/dynamic_tools.py` in the [GitHub repository](https://github.com/silvestrid/udspy/tree/main/examples)

---

## See Also

- [Dynamic Tools Guide](../examples/dynamic_tools.md) - Dynamic tool management
- [Confirmation API](confirmation.md) - Confirmation system and `@confirm_first` decorator
- [ReAct API](react.md) - Using tools with ReAct agents
- [ReAct Examples](../examples/react.md) - Tool usage examples
- [Module API](module.md) - Base module documentation
