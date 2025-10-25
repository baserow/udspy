# Confirmation System Architecture

The confirmation system provides human-in-the-loop (HITL) capabilities for udspy, allowing agents and tools to pause execution and request human approval, clarification, or feedback.

## Overview

The confirmation system is built on three core principles:

1. **Exception-based Flow Control**: Uses Python exceptions (`ConfirmationRequired`, `ConfirmationRejected`) to pause execution naturally
2. **Stateful Resumption**: Saves module state in the exception to enable resuming from the exact point where confirmation was requested
3. **Thread-safe Context**: Uses Python's `contextvars` for isolated, thread-safe confirmation tracking

## Core Components

### 1. Exceptions

#### `ConfirmationRequired`

Raised when human input is needed to proceed. Contains:

```python
class ConfirmationRequired(Exception):
    question: str              # Question to ask the user
    confirmation_id: str       # Unique ID for this confirmation
    tool_call: ToolCall | None # Optional tool call information
    context: dict[str, Any]    # Module-specific state for resumption
```

**Usage scenarios:**
- Tools decorated with `@confirm_first` before execution
- ReAct agent's `ask_to_user` tool for clarification
- Custom code needing human interaction

#### `ConfirmationRejected`

Raised when user explicitly rejects a confirmation. Distinguishes "user said no" from "pending approval".

```python
class ConfirmationRejected(Exception):
    message: str
    confirmation_id: str
    tool_call: ToolCall | None
```

### 2. Decorator: `@confirm_first`

Makes any function require confirmation before execution:

```python
from udspy import confirm_first

@confirm_first
def delete_database(name: str) -> str:
    # Dangerous operation
    return f"Deleted {name}"

# First call raises ConfirmationRequired
try:
    delete_database("production")
except ConfirmationRequired as e:
    # Get user approval
    respond_to_confirmation(e.confirmation_id, approved=True)

# Second call executes
result = delete_database("production")
```

**How it works:**
1. Generates stable confirmation ID from `function_name:hash(args)`
2. Checks confirmation context for approval status
3. If not approved: raises `ConfirmationRequired`
4. If approved: executes function and clears confirmation
5. If rejected: raises `ConfirmationRejected`

### 3. Confirmation Context

Thread-safe and async-safe state storage using `contextvars`:

```python
_confirmation_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "confirmation_context", default=None
)
```

Each confirmation is stored as:
```python
{
    "confirmation_id": {
        "approved": bool,
        "data": Any,  # Modified arguments if edited
        "status": str  # "pending", "approved", "rejected", "edited", "feedback"
    }
}
```

**Benefits:**
- Isolated per thread/async task
- No global state contamination
- Works in concurrent environments

### 4. State Management Functions

```python
# Set confirmation response
respond_to_confirmation(
    confirmation_id: str,
    approved: bool = True,
    data: Any = None,
    status: str | None = None
)

# Check confirmation status
status = get_confirmation_status(confirmation_id)  # "pending" | "approved" | "rejected"

# Cleanup
clear_confirmation(confirmation_id)
clear_all_confirmations()
```

## Integration with Modules

### Module Resume Pattern

Modules support two resumption patterns:

#### Pattern 1: Explicit `aresume()` Method

```python
try:
    result = await agent.aforward(question="Delete files")
except ConfirmationRequired as e:
    result = await agent.aresume("yes", e)
```

**Used when:**
- Need to handle confirmation differently than normal flow
- Want explicit control over resumption

#### Pattern 2: Loop-based with `resume_state`

```python
from udspy import ResumeState

resume_state = None

while True:
    try:
        result = await agent.aforward(
            question="Delete files",
            resume_state=resume_state
        )
        break
    except ConfirmationRequired as e:
        user_response = input(f"{e.question} (yes/no): ")
        resume_state = ResumeState(e, user_response)
```

**Used when:**
- Multiple confirmations may be needed
- Want uniform handling loop
- Building interactive CLIs or web APIs

### How Resume Works

When `resume_state` is provided:

1. `aforward()` detects `resume_state is not None`
2. Delegates to `aresume(user_response, resume_state)`
3. Module extracts saved state from exception's `.context`
4. Continues execution from saved point

**Example state in ReAct:**
```python
{
    "trajectory": {...},      # Reasoning/action history
    "iteration": 3,          # Which iteration to resume from
    "input_args": {...},     # Original input arguments
}
```

## Tool Confirmation Flow

### Creating Confirmable Tools

```python
from udspy import tool

@tool(name="delete_file", require_confirmation=True)
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"
```

Internally, this wraps the function with `@confirm_first`.

### Execution Flow

```
1. ReAct decides to call delete_file
   ↓
2. Tool.__call__() is invoked
   ↓
3. @confirm_first decorator checks context
   ↓
4. No approval found → raises ConfirmationRequired
   ↓
5. ReAct catches exception, saves state
   ↓
6. User responds via agent.aresume() or resume_state
   ↓
7. respond_to_confirmation() marks approved
   ↓
8. ReAct resumes with pending_tool_call
   ↓
9. @confirm_first sees approval, executes
   ↓
10. Result added to trajectory
```

## Design Rationale

### Why Exceptions?

**Alternatives considered:**
- **Callbacks**: More complex, harder to reason about control flow
- **Async generators with yield**: Breaks module composability
- **Return sentinel values**: Ambiguous, requires checking every return

**Why exceptions work:**
- Natural suspension of call stack
- Carries state in exception object
- Explicit handling with try/except
- Composes well with async/await

### Why Stable IDs?

Generated from `function_name:hash(args)` to enable:
- Same function call to resume after approval
- Idempotent confirmation (multiple attempts use same ID)
- Deterministic behavior for testing

### Why Two Exception Types?

- `ConfirmationRequired`: Suspends execution, waiting for response
- `ConfirmationRejected`: Terminates execution, user said "no"

Allows code to distinguish and handle differently:
```python
try:
    result = agent(question="Delete production data")
except ConfirmationRequired:
    # Still possible to proceed with approval
    pass
except ConfirmationRejected:
    # User explicitly denied - different handling
    log_denial()
```

## Thread Safety

### Guarantees

1. **Isolated contexts**: Each thread/task has its own confirmation state
2. **No race conditions**: `ContextVar` provides isolation
3. **Concurrent safe**: Multiple agents can run simultaneously

### Example

```python
import asyncio

async def run_agent(agent_id: int):
    agent = ReAct(...)
    try:
        return await agent.aforward(question=f"Task {agent_id}")
    except ConfirmationRequired as e:
        # Each agent has its own confirmation context
        respond_to_confirmation(e.confirmation_id, approved=True)
        return await agent.aforward(
            question=f"Task {agent_id}",
            resume_state=e,
            user_response="yes"
        )

# Run 10 agents concurrently - each isolated
results = await asyncio.gather(*[run_agent(i) for i in range(10)])
```

## Best Practices

### 1. Always Clean Up

Confirmations are auto-cleared on success, but clean up manually if needed:

```python
try:
    result = agent(question="...", resume_state=state)
except Exception:
    clear_all_confirmations()  # Reset on error
    raise
```

### 2. Limit Confirmation Rounds

Prevent infinite loops:

```python
from udspy import ResumeState

MAX_CONFIRMATIONS = 5
for attempt in range(MAX_CONFIRMATIONS):
    try:
        result = agent(question="...", resume_state=state)
        break
    except ConfirmationRequired as e:
        response = get_user_input(e.question)
        state = ResumeState(e, response)
else:
    raise RuntimeError("Too many confirmation requests")
```

### 3. Validate User Responses

```python
except ConfirmationRequired as e:
    response = input(f"{e.question} (yes/no/edit): ").lower()

    if response not in ("yes", "no", "edit"):
        print("Invalid response. Please enter yes, no, or edit.")
        continue

    state = e
    user_response = response
```

### 4. Provide Context to Users

```python
except ConfirmationRequired as e:
    print(f"\n⚠️  Confirmation Required")
    print(f"   Question: {e.question}")

    if e.tool_call:
        print(f"   Tool: {e.tool_call.name}")
        print(f"   Args: {json.dumps(e.tool_call.args, indent=2)}")

    response = input("\nYour response: ")
```

## Common Patterns

### Pattern: Web API with Session State

```python
# POST /agent/start
@app.post("/agent/start")
async def start_agent(request: AgentRequest):
    try:
        result = await agent.aforward(question=request.question)
        return {"status": "completed", "result": result.answer}
    except ConfirmationRequired as e:
        # Save state in session/DB
        session_id = save_state(e)
        return {
            "status": "confirmation_required",
            "session_id": session_id,
            "question": e.question,
            "tool_call": e.tool_call
        }

# POST /agent/resume
@app.post("/agent/resume")
async def resume_agent(request: ResumeRequest):
    from udspy import ResumeState

    exception = load_state(request.session_id)
    resume_state = ResumeState(exception, request.user_response)
    result = await agent.aforward(
        question=exception.context["original_question"],
        resume_state=resume_state
    )
    return {"status": "completed", "result": result.answer}
```

### Pattern: Interactive CLI

```python
from udspy import ResumeState

def run_interactive(agent, question):
    resume_state = None

    while True:
        try:
            result = agent(
                question=question,
                resume_state=resume_state
            )
            print(f"\n✓ {result.answer}")
            return result

        except ConfirmationRequired as e:
            print(f"\n⚠️  {e.question}")

            if e.tool_call:
                print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")

            response = input("\n[yes/no/edit]: ").strip()

            if response == "edit" and e.tool_call:
                print("Enter new args as JSON:")
                new_args = input("> ")
                resume_state = ResumeState(e, new_args)
            else:
                resume_state = ResumeState(e, response)

        except ConfirmationRejected as e:
            print(f"\n✗ Rejected: {e.message}")
            return None
```

### Pattern: Batch Processing with Selective Confirmation

```python
from udspy import ResumeState

async def process_batch(items, agent):
    results = []

    for item in items:
        resume_state = None
        attempts = 0

        while attempts < 3:
            try:
                result = await agent.aforward(
                    question=f"Process {item}",
                    resume_state=resume_state
                )
                results.append(result)
                break

            except ConfirmationRequired as e:
                # Auto-approve low-risk items
                if is_low_risk(e.tool_call):
                    response = "yes"
                else:
                    # Human review for high-risk
                    response = await get_human_approval(e)

                resume_state = ResumeState(e, response)
                attempts += 1

    return results
```

## See Also

- [Confirmation API Reference](../api/confirmation.md) - API documentation
- [Confirmation Examples](../examples/confirmation.md) - Practical examples
- [ReAct Module](modules/react.md) - ReAct integration with confirmations
- [Tool API](../api/tool.md) - Creating confirmable tools
