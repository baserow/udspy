# Interrupt API Reference

API documentation for the interrupt system that enables human-in-the-loop workflows.

## Module: `udspy.interrupt`

The interrupt system provides a general-purpose mechanism for pausing execution to request human input or approval. It's designed to be:
- **Thread-safe**: Works correctly with multi-threaded applications
- **Task-safe**: Compatible with asyncio concurrent tasks
- **Module-agnostic**: Can be used by any module, not just ReAct

### `HumanInTheLoopRequired`

```python
class HumanInTheLoopRequired(Exception):
    """Raised when human input is needed to proceed."""
```

Exception that pauses execution and saves state for resumption. This is the core mechanism for implementing human-in-the-loop workflows.

#### Constructor

```python
def __init__(
    self,
    question: str,
    *,
    interrupt_id: str | None = None,
    tool_call: ToolCall | None = None,
    context: dict[str, Any] | None = None,
)
```

**Parameters:**

- **`question`** (`str`): The question to ask the user
  - Should be clear and actionable
  - Example: "Confirm execution of delete_file with args: {'path': '/tmp/test.txt'}?"
- **`interrupt_id`** (`str | None`, optional): Unique interrupt identifier
  - Auto-generated UUID if not provided
  - Used to track interrupt status
- **`tool_call`** (`ToolCall | None`, optional): Information about the tool call that triggered this interrupt
  - Contains tool name, arguments, and optional call ID
  - Can be `None` for non-tool interrupts
- **`context`** (`dict[str, Any] | None`, optional): Module-specific state dictionary
  - Used to save execution state for resumption
  - Each module defines its own context structure

**Example:**

```python
from udspy.interrupt import HumanInTheLoopRequired, ToolCall

# Simple interrupt with just a question
raise HumanInTheLoopRequired("Do you want to proceed?")

# Interrupt with tool call information
raise HumanInTheLoopRequired(
    question="Confirm deletion?",
    tool_call=ToolCall(
        name="delete_file",
        args={"path": "/tmp/test.txt"}
    )
)

# Interrupt with module state
raise HumanInTheLoopRequired(
    question="Need clarification",
    context={
        "iteration": 5,
        "trajectory": {...},
        "input_args": {...}
    }
)
```

#### Attributes

##### `question`

```python
question: str
```

The question being asked to the user.

##### `interrupt_id`

```python
interrupt_id: str
```

Unique identifier for this interrupt. Use with `get_interrupt_status()` and `set_interrupt_approval()`.

##### `tool_call`

```python
tool_call: ToolCall | None
```

Optional tool call information. See `ToolCall` class below.

##### `context`

```python
context: dict[str, Any]
```

Module-specific state dictionary. Structure depends on the module that raised the exception.

---

### `ToolCall`

```python
class ToolCall:
    """Information about a tool call that triggered an interrupt."""
```

Encapsulates information about a tool invocation.

#### Constructor

```python
def __init__(
    self,
    name: str,
    args: dict[str, Any],
    call_id: str | None = None
)
```

**Parameters:**

- **`name`** (`str`): Tool name
- **`args`** (`dict[str, Any]`): Tool arguments as keyword arguments
- **`call_id`** (`str | None`, optional): Call ID from the LLM provider (e.g., OpenAI)

**Example:**

```python
from udspy.interrupt import ToolCall

tool_call = ToolCall(
    name="search",
    args={"query": "Python tutorials"},
    call_id="call_abc123"
)
```

#### Attributes

##### `name`

```python
name: str
```

The name of the tool.

##### `args`

```python
args: dict[str, Any]
```

The tool arguments as a dictionary.

##### `call_id`

```python
call_id: str | None
```

Optional call ID from the LLM provider.

---

### `@interruptible`

```python
def interruptible(func: Callable) -> Callable:
    """Decorator that makes a function require approval before execution."""
```

Decorator that wraps a function to require human approval on first call. Subsequent calls with the same arguments proceed normally after approval.

**How it works:**

1. First call: Raises `HumanInTheLoopRequired` with tool call information
2. User approves: Call `set_interrupt_approval(interrupt_id, approved=True)`
3. Subsequent calls: Execute normally if approved

**Supports:**
- Sync and async functions
- Positional and keyword arguments
- Thread-safe and asyncio task-safe execution

**Example:**

```python
from udspy.interrupt import interruptible, HumanInTheLoopRequired, set_interrupt_approval

@interruptible
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

# First call raises
try:
    delete_file("/tmp/test.txt")
except HumanInTheLoopRequired as e:
    print(e.question)  # "Confirm execution of delete_file..."
    interrupt_id = e.interrupt_id

    # User approves
    set_interrupt_approval(interrupt_id, approved=True)

    # Second call succeeds
    result = delete_file("/tmp/test.txt")
    print(result)  # "Deleted /tmp/test.txt"
```

**With async functions:**

```python
@interruptible
async def async_delete(path: str) -> str:
    await asyncio.sleep(0.1)
    os.remove(path)
    return f"Deleted {path}"

try:
    await async_delete("/tmp/test.txt")
except HumanInTheLoopRequired as e:
    set_interrupt_approval(e.interrupt_id, approved=True)
    result = await async_delete("/tmp/test.txt")
```

**Modifying arguments:**

```python
try:
    delete_file("/tmp/test.txt")
except HumanInTheLoopRequired as e:
    # Approve with modified arguments
    modified_args = {"path": "/tmp/safe.txt"}
    set_interrupt_approval(e.interrupt_id, approved=True, data=modified_args)

    # Next call uses modified args
    result = delete_file("/tmp/test.txt")
    print(result)  # "Deleted /tmp/safe.txt"
```

---

### `get_interrupt_status()`

```python
def get_interrupt_status(interrupt_id: str) -> str | None:
    """Get the status of an interrupt."""
```

Returns the current status of an interrupt by its ID.

**Parameters:**

- **`interrupt_id`** (`str`): The interrupt ID to query

**Returns:**

- `str | None`: One of:
  - `"pending"`: No decision made yet (or ID not found)
  - `"approved"`: User approved the action
  - `"rejected"`: User rejected the action
  - `"edited"`: User approved with modifications
  - `"feedback"`: User provided feedback for the agent

**Example:**

```python
from udspy.interrupt import get_interrupt_status, HumanInTheLoopRequired

try:
    agent(question="Delete files")
except HumanInTheLoopRequired as e:
    status = get_interrupt_status(e.interrupt_id)
    print(status)  # "pending"

    # After user responds
    agent.resume("yes", e)
    status = get_interrupt_status(e.interrupt_id)
    print(status)  # "approved"
```

---

### `set_interrupt_approval()`

```python
def set_interrupt_approval(
    interrupt_id: str,
    approved: bool = True,
    data: Any = None,
    status: str | None = None
) -> None:
    """Mark an interrupt as approved or rejected."""
```

Sets the approval status for an interrupt, optionally providing modified data.

**Parameters:**

- **`interrupt_id`** (`str`): The interrupt ID to update
- **`approved`** (`bool`, default: `True`): Whether to approve or reject
  - `True`: Allow execution to proceed
  - `False`: Block execution
- **`data`** (`Any`, optional): Modified arguments or feedback data
  - For `@interruptible` functions: Dict with modified arguments
  - For modules: Any data to pass back
- **`status`** (`str | None`, optional): Explicit status to set
  - If not provided, inferred from `approved` and `data`
  - Can be: "approved", "rejected", "edited", "feedback"

**Example:**

```python
from udspy.interrupt import set_interrupt_approval

# Simple approval
set_interrupt_approval("abc-123", approved=True)

# Rejection
set_interrupt_approval("abc-123", approved=False)

# Approval with modified arguments
set_interrupt_approval(
    "abc-123",
    approved=True,
    data={"path": "/safe/location.txt"}
)

# Explicit status
set_interrupt_approval(
    "abc-123",
    approved=True,
    status="feedback"
)
```

---

### `get_interrupt_context()`

```python
def get_interrupt_context() -> dict[str, dict[str, Any]]:
    """Get the current interrupt context dictionary."""
```

Returns the complete interrupt context for the current thread/task. Mostly used for debugging.

**Returns:**

- `dict[str, dict[str, Any]]`: Dictionary mapping interrupt IDs to their state:
  ```python
  {
      "interrupt-id-1": {
          "approved": True,
          "data": {...},
          "status": "approved"
      },
      "interrupt-id-2": {
          "approved": False,
          "status": "rejected"
      }
  }
  ```

**Example:**

```python
from udspy.interrupt import get_interrupt_context

context = get_interrupt_context()
print(f"Active interrupts: {len(context)}")
for interrupt_id, state in context.items():
    print(f"{interrupt_id}: {state['status']}")
```

---

### `clear_interrupt()`

```python
def clear_interrupt(interrupt_id: str) -> None:
    """Remove an interrupt from the context."""
```

Clears a specific interrupt from the context. Usually done automatically after successful execution.

**Parameters:**

- **`interrupt_id`** (`str`): The interrupt ID to clear

**Example:**

```python
from udspy.interrupt import clear_interrupt

clear_interrupt("abc-123")
```

---

### `clear_all_interrupts()`

```python
def clear_all_interrupts() -> None:
    """Clear all interrupts from the context."""
```

Removes all interrupts from the current context. Useful for cleanup or testing.

**Example:**

```python
from udspy.interrupt import clear_all_interrupts

# Start fresh
clear_all_interrupts()
```

---

## Interrupt Status Lifecycle

The status of an interrupt follows this lifecycle:

```
pending (initial)
    ↓
    ├→ approved (user said "yes")
    ├→ rejected (user said "no")
    ├→ edited (user modified args)
    └→ feedback (user provided feedback)
```

**Status Meanings:**

- **pending**: Initial state, no decision made
- **approved**: User approved the action as-is
- **rejected**: User rejected the action
- **edited**: User approved with modifications to arguments
- **feedback**: User provided textual feedback (not yes/no)

---

## Thread Safety

The interrupt system uses `contextvars.ContextVar` for thread-safe and asyncio task-safe storage:

- Each thread has its own interrupt context
- Each asyncio task inherits parent task's context
- No cross-contamination between threads/tasks

**Example:**

```python
import threading
from udspy.interrupt import interruptible, HumanInTheLoopRequired, set_interrupt_approval

@interruptible
def thread_func(thread_id: int) -> str:
    return f"Thread {thread_id}"

def worker(thread_id: int):
    try:
        thread_func(thread_id)
    except HumanInTheLoopRequired as e:
        # Each thread has its own interrupt context
        set_interrupt_approval(e.interrupt_id, approved=True)
        result = thread_func(thread_id)
        print(result)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Integration with Modules

Modules can use the interrupt system by:

1. Raising `HumanInTheLoopRequired` when human input is needed
2. Saving state in the `context` dict
3. Implementing `resume()` and `aresume()` methods to restore state

**Example Module:**

```python
from udspy import Module, Prediction
from udspy.interrupt import HumanInTheLoopRequired

class MyModule(Module):
    def forward(self, input: str) -> Prediction:
        # ... some work ...

        if needs_human_input:
            raise HumanInTheLoopRequired(
                question="Please confirm",
                context={
                    "current_step": 5,
                    "partial_result": "...",
                    "input": input
                }
            )

        # ... continue work ...
        return Prediction(output="result")

    def resume(self, user_response: str, saved_state: HumanInTheLoopRequired) -> Prediction:
        # Restore state from context
        current_step = saved_state.context["current_step"]
        partial_result = saved_state.context["partial_result"]
        input = saved_state.context["input"]

        # Process user response
        if user_response.lower() == "yes":
            # Continue from where we left off
            pass

        # ... complete work ...
        return Prediction(output="final result")
```

---

## Integration with Tools

Tools can use `interruptible=True` parameter to require confirmation:

```python
from udspy import tool
from pydantic import Field

@tool(
    name="delete_file",
    description="Delete a file",
    interruptible=True  # Wraps function with @interruptible decorator
)
def delete_file(path: str = Field(...)) -> str:
    os.remove(path)
    return f"Deleted {path}"
```

The `@tool` decorator automatically wraps the function with `@interruptible` when this parameter is set.

---

## See Also

- [ReAct API](react.md) - ReAct agent that uses the interrupt system
- [Tool API](tool.md) - Creating interruptible tools
- [Module API](module.md) - Base module with suspend/resume methods
