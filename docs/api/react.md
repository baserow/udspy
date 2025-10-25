# ReAct API Reference

API documentation for the ReAct (Reasoning and Acting) module.

## Module: `udspy.module.react`

### `ReAct`

```python
class ReAct(Module):
    """ReAct (Reasoning and Acting) module for tool-using agents."""
```

Agent module that iteratively reasons about the current situation and decides whether to call a tool or finish the task.

#### Constructor

```python
def __init__(
    self,
    signature: type[Signature] | str,
    tools: list[Callable | Tool],
    *,
    max_iters: int = 10,
    enable_ask_to_user: bool = True,
)
```

**Parameters:**

- **`signature`** (`type[Signature] | str`): Task signature defining inputs and outputs
  - Can be a `Signature` class or a string like `"input1, input2 -> output1, output2"`
- **`tools`** (`list[Callable | Tool]`): List of tool functions or `Tool` objects
  - Tools can be decorated functions (`@tool`) or `Tool` instances
- **`max_iters`** (`int`, default: `10`): Maximum number of reasoning iterations
  - Agent will stop after this many steps even if not finished
- **`enable_ask_to_user`** (`bool`, default: `True`): Whether to enable the `ask_to_user` tool
  - If `False`, the agent cannot request user clarification

**Example:**

```python
from udspy import ReAct, Signature, InputField, OutputField, tool
from pydantic import Field

@tool(name="search", description="Search for information")
def search(query: str = Field(...)) -> str:
    return f"Results for: {query}"

class QA(Signature):
    """Answer questions using available tools."""
    question: str = InputField()
    answer: str = OutputField()

agent = ReAct(QA, tools=[search], max_iters=10)
```

#### Methods

##### `forward(**input_args) -> Prediction`

Synchronous forward pass through the ReAct loop.

**Parameters:**

- **`**input_args`**: Input values matching the signature's input fields
  - Keys must match input field names
  - Can include `max_iters` to override default

**Returns:**

- `Prediction`: Contains output fields and trajectory
  - Trajectory tracks all reasoning steps
  - Output fields match the signature

**Raises:**

- `ConfirmationRequired`: When user input is needed
  - Raised when `ask_to_user` is called
  - Raised when tool requires confirmation
  - Contains saved state for resumption

**Example:**

```python
result = agent(question="What is Python?")
print(result.answer)
print(result.trajectory)  # All reasoning steps
```

##### `aforward(**input_args) -> Prediction`

Async forward pass through the ReAct loop.

**Parameters:**

- Same as `forward()`

**Returns:**

- `Prediction`: Same as `forward()`

**Raises:**

- Same as `forward()`

**Example:**

```python
import asyncio

async def main():
    result = await agent.aforward(question="What is Python?")
    print(result.answer)

asyncio.run(main())
```

##### `resume(user_response, saved_state) -> Prediction`

Resume execution after user provides input (synchronous).

**Parameters:**

- **`user_response`** (`str`): The user's response to the question
  - `"yes"` or `"y"`: Approve the pending tool call
  - `"no"` or `"n"`: Reject and let agent decide next action
  - Free text: Treated as feedback for the agent
  - JSON with `"edit"` key: Modify tool name/args (e.g., `{"edit": {"name": "new_tool", "args": {...}}}`)
- **`saved_state`** (`ConfirmationRequired`): The exception that was raised
  - Contains `context` dict with: `trajectory`, `iteration`, `input_args`
  - Contains `tool_call` with pending tool information
  - Contains `confirmation_id` for tracking

**Returns:**

- `Prediction`: Final result after resuming execution

**Example:**

```python
from udspy import ConfirmationRequired

try:
    result = agent(question="Delete my files")
except ConfirmationRequired as e:
    print(f"Agent asks: {e.question}")
    if e.tool_call:
        print(f"Tool: {e.tool_call.name}")
        print(f"Args: {e.tool_call.args}")
    response = input("Your response (yes/no/feedback): ")
    result = agent.resume(response, e)
```

##### `aresume(user_response, saved_state) -> Prediction`

Resume execution after user provides input (async).

**Parameters:**

- Same as `resume()`

**Returns:**

- Same as `resume()`

**Example:**

```python
try:
    result = await agent.aforward(question="Delete my files")
except ConfirmationRequired as e:
    print(f"Agent asks: {e.question}")
    response = get_user_input(e.question)
    result = await agent.aresume(response, e)
```

#### Properties

##### `signature`

```python
signature: type[Signature]
```

The task signature used by this agent.

##### `tools`

```python
tools: dict[str, Tool]
```

Dictionary mapping tool names to Tool objects. Includes:
- User-provided tools
- Built-in `finish` tool
- Built-in `ask_to_user` tool (if enabled)

##### `react_signature`

```python
react_signature: type[Signature]
```

Internal signature for the reasoning loop. With native tool calling:
- Outputs `reasoning`: The agent's reasoning about what to do next
- Tools are called natively via OpenAI's tool calling API
- The LLM both produces reasoning text and selects tools simultaneously

##### `extract_signature`

```python
extract_signature: type[Signature]
```

Internal signature for extracting the final answer from the trajectory.

---

### `ConfirmationRequired`

```python
class ConfirmationRequired(Exception):
    """Raised when human input is needed to proceed."""
```

**Note**: This exception has been moved to the `confirmation` module. See the [Confirmation API](confirmation.md) for full documentation.

Exception that pauses ReAct execution and saves state for resumption. This exception can be raised by:
- The `ask_to_user` tool when the agent needs clarification
- Tools with `require_confirmation=True` before execution
- Custom tools that need human input

#### Constructor

```python
from udspy.confirmation import ConfirmationRequired, ToolCall

def __init__(
    self,
    question: str,
    *,
    confirmation_id: str | None = None,
    tool_call: ToolCall | None = None,
    context: dict[str, Any] | None = None,
)
```

**Parameters:**

- **`question`** (`str`): Question to ask the user
- **`confirmation_id`** (`str | None`): Unique confirmation ID (auto-generated if not provided)
- **`tool_call`** (`ToolCall | None`): Optional tool call information
  - Has attributes: `.name` (tool name), `.args` (arguments dict), `.call_id` (optional ID)
- **`context`** (`dict[str, Any] | None`): Module-specific state dictionary
  - ReAct stores: `trajectory`, `iteration`, `input_args` here

#### Attributes

##### `question`

```python
question: str
```

The question being asked to the user.

**Example:**

```python
try:
    result = agent(question="Delete files")
except ConfirmationRequired as e:
    print(e.question)  # "Confirm execution of delete_file...?"
```

##### `confirmation_id`

```python
confirmation_id: str
```

Unique identifier for this confirmation. Used with `get_confirmation_status()` to check approval status.

**Example:**

```python
from udspy import get_confirmation_status

try:
    result = agent(question="Delete files")
except ConfirmationRequired as e:
    print(e.confirmation_id)  # "abc-123-def-456"
    status = get_confirmation_status(e.confirmation_id)
    print(status)  # "pending"
```

##### `tool_call`

```python
tool_call: ToolCall | None
```

Information about the tool call that triggered this confirmation (if applicable).

**Attributes:**
- `name` (`str`): Tool name
- `args` (`dict[str, Any]`): Tool arguments
- `call_id` (`str | None`): Optional call ID from OpenAI

**Example:**

```python
try:
    result = agent(question="Delete my files")
except ConfirmationRequired as e:
    if e.tool_call:
        print(f"Tool: {e.tool_call.name}")  # "delete_file"
        print(f"Args: {e.tool_call.args}")  # {"path": "/tmp/test.txt"}
        print(f"Call ID: {e.tool_call.call_id}")  # "call_abc123"
```

##### `context`

```python
context: dict[str, Any]
```

Module-specific state dictionary. For ReAct agents, contains:
- `trajectory` (`dict[str, Any]`): Current execution trajectory with reasoning, tool calls, and observations
- `iteration` (`int`): Current iteration number (0-indexed)
- `input_args` (`dict[str, Any]`): Original input arguments

**Example:**

```python
try:
    result = agent(question="What is 2+2?")
except ConfirmationRequired as e:
    # Access ReAct-specific state
    trajectory = e.context["trajectory"]
    print(trajectory)
    # {
    #   'reasoning_0': 'I should use the calculator',
    #   'tool_name_0': 'calculator',
    #   'tool_args_0': {'expression': '2+2'},
    #   'observation_0': '4'
    # }

    iteration = e.context["iteration"]
    print(f"Paused at step {iteration + 1}")

    input_args = e.context["input_args"]
    print(f"Original question: {input_args['question']}")
```

---

## Built-in Tools

Every ReAct agent automatically includes these tools:

### `finish`

Tool that signals task completion.

**Name:** `finish`

**Description:** "Call this when you have all information needed to produce {outputs}"

**Arguments:** None

**Usage:**

The agent automatically selects this tool when it has enough information to answer. You don't call it directly.

### `ask_to_user`

Tool for requesting user clarification (if enabled).

**Name:** `ask_to_user`

**Description:** "Ask the user for clarification. ONLY use this when: ..."

**Arguments:**
- `question` (`str`): The question to ask the user

**Usage:**

The agent can call this whenever it needs clarification or more information from the user.

Raises `ConfirmationRequired` exception.

**Notes:**
- Can be disabled with `enable_ask_to_user=False`

---

## Trajectory Format

The trajectory is a dictionary with the following keys:

```python
{
    "reasoning_0": str,    # Agent's reasoning for step 0
    "tool_name_0": str,    # Tool name selected for step 0
    "tool_args_0": dict,   # Arguments for step 0
    "observation_0": str,  # Tool result for step 0

    "reasoning_1": str,    # Agent's reasoning for step 1
    "tool_name_1": str,    # Tool name selected for step 1
    "tool_args_1": dict,   # Arguments for step 1
    "observation_1": str,  # Tool result for step 1

    # ... continues for all iterations
}
```

**Example:**

```python
result = agent(question="Calculate 2+2")

# Access trajectory
print(result.trajectory)
# {
#     'reasoning_0': 'I need to calculate 2+2',
#     'tool_name_0': 'calculator',
#     'tool_args_0': {'expression': '2+2'},
#     'observation_0': '4',
#     'reasoning_1': 'I have the answer',
#     'tool_name_1': 'finish',
#     'tool_args_1': {},
#     'observation_1': 'Task completed'
# }

# Iterate through steps
i = 0
while f"observation_{i}" in result.trajectory:
    print(f"Step {i}:")
    print(f"  Reasoning: {result.trajectory.get(f'reasoning_{i}', '')}")
    print(f"  Tool: {result.trajectory[f'tool_name_{i}']}")
    print(f"  Args: {result.trajectory[f'tool_args_{i}']}")
    print(f"  Result: {result.trajectory[f'observation_{i}']}")
    i += 1
```

---

## String Signature Format

For quick prototyping, you can use string signatures:

**Format:** `"input1, input2 -> output1, output2"`

**Examples:**

```python
# Single input, single output
agent = ReAct("query -> result", tools=[search])

# Multiple inputs
agent = ReAct("context, question -> answer", tools=[search])

# Multiple outputs
agent = ReAct("topic -> summary, sources", tools=[search])
```

The string signature is parsed into:
- Input fields: All fields before `->` (type: `str`)
- Output fields: All fields after `->` (type: `str`)

---

## Tool Confirmation

Tools can require user confirmation before execution using the `require_confirmation` parameter:

```python
from udspy import tool
from pydantic import Field

@tool(
    name="delete_file",
    description="Delete a file",
    require_confirmation=True  # Require confirmation before execution
)
def delete_file(path: str = Field(...)) -> str:
    os.remove(path)
    return f"Deleted {path}"
```

When the agent tries to call this tool, it raises `ConfirmationRequired` with a confirmation question on the first call. After the user approves, the tool executes normally.

**Confirmation Message Format:**

```
"Confirm execution of {tool_name} with args: {args}? (yes/no)"
```

**Response Options:**
- `"yes"` or `"y"`: Approve and execute the tool
- `"no"` or `"n"`: Reject and let agent choose a different action
- JSON with `"edit"`: Modify tool arguments before execution

**Example:**

```python
try:
    result = agent(question="Delete all temporary files")
except ConfirmationRequired as e:
    print(e.question)  # "Confirm execution of delete_file..."
    # User approves
    result = agent.resume("yes", e)
```

---

## Error Handling

### Tool Execution Errors

If a tool raises an exception, the error is captured as an observation:

```python
@tool(name="api_call", description="Call API")
def api_call(endpoint: str = Field(...)) -> str:
    if endpoint == "invalid":
        raise ValueError("Invalid endpoint")
    return "Success"

# Agent will see observation:
# "Error executing api_call: Invalid endpoint"
```

The agent can then:
1. Try a different tool
2. Retry with different arguments
3. Ask the user for help (using the `ask_to_user` tool)

### Maximum Iterations

If the agent reaches `max_iters`, it stops and extracts an answer from the current trajectory:

```python
agent = ReAct(signature, tools=tools, max_iters=5)
result = agent(question="Complex task")
# Will stop after 5 iterations even if not finished
```

---

## Type Annotations

```python
from typing import Callable, Any
from udspy import ReAct, Signature, Tool, Prediction, ConfirmationRequired

# Constructor types
signature: type[Signature] | str
tools: list[Callable | Tool]
max_iters: int
enable_ask_to_user: bool

# Method types
def forward(**input_args: Any) -> Prediction: ...
async def aforward(**input_args: Any) -> Prediction: ...
def resume(
    user_response: str,
    saved_state: ConfirmationRequired
) -> Prediction: ...
async def aresume(
    user_response: str,
    saved_state: ConfirmationRequired
) -> Prediction: ...
```

---

## See Also

- [ReAct Examples](../examples/react.md) - Usage guide and examples
- [Confirmation API](confirmation.md) - Confirmation system and `ConfirmationRequired` documentation
- [Tool API](tool.md) - Creating and configuring tools
- [Module API](module.md) - Base module documentation
- [Signature API](signature.md) - Signature documentation
