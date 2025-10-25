# Confirmation Examples

This guide provides practical examples of using udspy's confirmation system for human-in-the-loop workflows.

## Quick Start

### Basic Confirmation Loop

The simplest pattern for handling confirmations is a loop with try/except:

```python
from udspy import ReAct, ConfirmationRequired, ResumeState, tool
from pydantic import Field

# Define a tool that requires confirmation
@tool(name="delete_file", require_confirmation=True)
def delete_file(path: str = Field(description="File path")) -> str:
    import os
    os.remove(path)
    return f"Deleted {path}"

# Create agent
agent = ReAct("question -> answer", tools=[delete_file])

# Handle confirmations in a loop
resume_state = None

while True:
    try:
        result = agent(
            question="Delete the file /tmp/old_data.txt",
            resume_state=resume_state
        )
        print(f"Success: {result.answer}")
        break

    except ConfirmationRequired as e:
        print(f"Confirmation needed: {e.question}")
        response = input("Your response (yes/no): ")
        resume_state = ResumeState(e, response)
```

## Interactive CLI Examples

### Example 1: Basic Interactive Loop

```python
import asyncio
from udspy import ReAct, ConfirmationRequired, ResumeState, ConfirmationRejected, tool
from pydantic import Field

@tool(name="delete_file", require_confirmation=True)
def delete_file(path: str = Field(description="File path")) -> str:
    return f"Would delete {path}"

@tool(name="search")
def search(query: str = Field(description="Search query")) -> str:
    return f"Results for: {query}"

async def interactive_agent():
    """Run agent with interactive confirmation handling."""
    agent = ReAct(
        "task -> result",
        tools=[search, delete_file],
        max_iters=5
    )

    task = "Search for Python tutorials and delete /tmp/cache.txt"
    resume_state = None

    while True:
        try:
            result = await agent.aforward(
                task=task,
                resume_state=resume_state
            )
            print(f"\n✓ Task completed!")
            print(f"  Result: {result.result}")
            return result

        except ConfirmationRequired as e:
            print(f"\n⚠️  Confirmation Required")
            print(f"   {e.question}")

            if e.tool_call:
                print(f"   Tool: {e.tool_call.name}")
                print(f"   Args: {e.tool_call.args}")

            response = input("\n   Response (yes/no): ").strip()
            user_response = response
            resume_state = ResumeState(e, response)

        except ConfirmationRejected as e:
            print(f"\n✗ Operation rejected: {e.message}")
            return None

if __name__ == "__main__":
    asyncio.run(interactive_agent())
```

### Example 2: Enhanced Interactive Loop with Argument Editing

```python
import json
from udspy import ReAct, ConfirmationRequired, ResumeState, tool
from pydantic import Field

@tool(name="write_file", require_confirmation=True)
def write_file(
    path: str = Field(description="File path"),
    content: str = Field(description="File content")
) -> str:
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"

def run_with_editing(agent, question):
    """Interactive loop with argument editing support."""
    resume_state = None

    while True:
        try:
            result = agent(
                question=question,
                resume_state=resume_state
            )
            print(f"\n✓ Success: {result.answer}")
            return result

        except ConfirmationRequired as e:
            print(f"\n⚠️  Confirmation Required")
            print(f"   {e.question}")

            if e.tool_call:
                print(f"\n   Tool: {e.tool_call.name}")
                print(f"   Args:")
                for key, value in e.tool_call.args.items():
                    print(f"     {key}: {value}")

            print("\n   Options:")
            print("   - 'yes' to approve")
            print("   - 'no' to reject")
            print("   - 'edit' to modify arguments")

            response = input("\n   Your choice: ").strip().lower()

            if response == "edit" and e.tool_call:
                # Let user edit arguments
                print("\n   Current arguments:")
                print(f"   {json.dumps(e.tool_call.args, indent=2)}")
                print("\n   Enter new arguments as JSON:")
                new_args_str = input("   > ")

                try:
                    new_args = json.loads(new_args_str)
                    user_response = json.dumps(new_args)
                except json.JSONDecodeError:
                    print("   Invalid JSON, treating as feedback")
                    user_response = new_args_str
            else:
                user_response = response

            resume_state = ResumeState(e, response)

# Usage
agent = ReAct(
    "task -> answer",
    tools=[write_file],
    max_iters=5
)

result = run_with_editing(
    agent,
    "Write 'Hello World' to /tmp/greeting.txt"
)
```

### Example 3: Helper Function for Common Pattern

```python
from typing import Callable
from udspy import Module, ConfirmationRequired, ResumeState, ConfirmationRejected

def run_with_confirmations(
    agent: Module,
    max_confirmations: int = 10,
    on_confirmation: Callable[[ConfirmationRequired], str] | None = None,
    **inputs
):
    """
    Helper function to run an agent with automatic confirmation handling.

    Args:
        agent: The agent/module to run
        max_confirmations: Maximum number of confirmation rounds
        on_confirmation: Optional callback for handling confirmations.
                        If None, uses default CLI input.
        **inputs: Input arguments for the agent

    Returns:
        The final prediction result

    Raises:
        RuntimeError: If max_confirmations is exceeded
        ConfirmationRejected: If user rejects the operation
    """
    resume_state = None

    for attempt in range(max_confirmations):
        try:
            return agent(
                resume_state=resume_state,
                **inputs
            )

        except ConfirmationRequired as e:
            if on_confirmation:
                # Use custom handler
                user_response = on_confirmation(e)
            else:
                # Default CLI handler
                print(f"\n⚠️  {e.question}")
                if e.tool_call:
                    print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")
                response = input("   Response (yes/no): ").strip()

            resume_state = ResumeState(e, response)

    raise RuntimeError(f"Exceeded {max_confirmations} confirmation requests")

# Usage example 1: Simple CLI
agent = ReAct("task -> result", tools=[delete_file])
result = run_with_confirmations(agent, task="Delete old logs")

# Usage example 2: Custom handler
def auto_approve_safe(e: ConfirmationRequired) -> str:
    # Auto-approve if path is in /tmp
    if e.tool_call and "/tmp" in str(e.tool_call.args.get("path", "")):
        print(f"Auto-approved: {e.tool_call.name}")
        return "yes"
    # Otherwise ask user
    print(f"Manual review needed: {e.question}")
    return input("Approve? (yes/no): ")

result = run_with_confirmations(
    agent,
    task="Delete /tmp/cache and /important/data",
    on_confirmation=auto_approve_safe
)
```

## Web API Examples

### Example 4: FastAPI with Session State

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from udspy import ReAct, ConfirmationRequired, ResumeState, tool
import uuid

app = FastAPI()

# In-memory session storage (use Redis/DB in production)
sessions: Dict[str, Dict[str, Any]] = {}

class StartRequest(BaseModel):
    question: str

class ResumeRequest(BaseModel):
    session_id: str
    user_response: str

class StartResponse(BaseModel):
    status: str
    session_id: str | None = None
    question: str | None = None
    tool_call: Dict | None = None
    result: str | None = None

@tool(name="delete_file", require_confirmation=True)
def delete_file(path: str) -> str:
    return f"Deleted {path}"

# Create agent
agent = ReAct("question -> answer", tools=[delete_file])

@app.post("/agent/start", response_model=StartResponse)
async def start_agent(request: StartRequest):
    """Start a new agent execution."""
    try:
        result = await agent.aforward(question=request.question)
        return StartResponse(
            status="completed",
            result=result.answer
        )

    except ConfirmationRequired as e:
        # Save state in session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "state": e,
            "question": request.question
        }

        return StartResponse(
            status="confirmation_required",
            session_id=session_id,
            question=e.question,
            tool_call=e.tool_call.__dict__ if e.tool_call else None
        )

@app.post("/agent/resume", response_model=StartResponse)
async def resume_agent(request: ResumeRequest):
    """Resume agent execution after confirmation."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[request.session_id]

    try:
        result = await agent.aforward(
            question=session["question"],
            resume_state=session["state"],
            user_response=request.user_response
        )

        # Clean up session on success
        del sessions[request.session_id]

        return StartResponse(
            status="completed",
            result=result.answer
        )

    except ConfirmationRequired as e:
        # Need another confirmation - update session
        session["state"] = e

        return StartResponse(
            status="confirmation_required",
            session_id=request.session_id,
            question=e.question,
            tool_call=e.tool_call.__dict__ if e.tool_call else None
        )
```

### Example 5: Async Queue Pattern

```python
import asyncio
from asyncio import Queue
from typing import Any
from udspy import ReAct, ConfirmationRequired, ResumeState

class ConfirmationRequest:
    def __init__(self, exception: ConfirmationRequired):
        self.exception = exception
        self.response_future = asyncio.Future()

    async def get_response(self) -> str:
        return await self.response_future

    def set_response(self, response: str):
        self.response_future.set_result(response)

async def agent_worker(
    agent: ReAct,
    question: str,
    confirmation_queue: Queue[ConfirmationRequest]
):
    """Worker that processes tasks and sends confirmations to queue."""
    resume_state = None

    while True:
        try:
            result = await agent.aforward(
                question=question,
                resume_state=resume_state
            )
            return result

        except ConfirmationRequired as e:
            # Send confirmation request to queue
            req = ConfirmationRequest(e)
            await confirmation_queue.put(req)

            # Wait for user response
            user_response = await req.get_response()
            resume_state = ResumeState(e, response)

async def confirmation_handler(confirmation_queue: Queue[ConfirmationRequest]):
    """Handler that processes confirmation requests from queue."""
    while True:
        req = await confirmation_queue.get()
        e = req.exception

        print(f"\n⚠️  Confirmation: {e.question}")
        if e.tool_call:
            print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")

        # Get user input (in real app, might be from WebSocket, UI, etc.)
        response = input("   Response: ")
        req.set_response(response)

async def run_with_queue(agent, question):
    """Run agent with async confirmation handling."""
    queue = Queue()

    # Start both worker and handler
    worker_task = asyncio.create_task(agent_worker(agent, question, queue))
    handler_task = asyncio.create_task(confirmation_handler(queue))

    try:
        result = await worker_task
        handler_task.cancel()
        return result
    except Exception as e:
        handler_task.cancel()
        raise
```

## Advanced Patterns

### Example 6: Multi-Agent with Shared Confirmations

```python
import asyncio
from typing import List
from udspy import ReAct, ConfirmationRequired, ResumeState

async def run_agent_with_confirmations(
    agent: ReAct,
    agent_id: int,
    question: str
) -> Any:
    """Run single agent with confirmation handling."""
    resume_state = None

    for attempt in range(5):
        try:
            result = await agent.aforward(
                question=question,
                resume_state=resume_state
            )
            return {
                "agent_id": agent_id,
                "result": result.answer,
                "status": "success"
            }

        except ConfirmationRequired as e:
            print(f"\n[Agent {agent_id}] {e.question}")
            response = input(f"[Agent {agent_id}] Response: ")
            resume_state = ResumeState(e, response)

    return {
        "agent_id": agent_id,
        "result": None,
        "status": "too_many_confirmations"
    }

async def run_multi_agent(questions: List[str]):
    """Run multiple agents concurrently with interactive confirmations."""
    agent = ReAct("question -> answer", tools=[...])

    tasks = [
        run_agent_with_confirmations(agent, i, q)
        for i, q in enumerate(questions)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Usage
questions = [
    "Analyze log files",
    "Clean up temp directory",
    "Generate report"
]

results = asyncio.run(run_multi_agent(questions))
```

### Example 7: Conditional Auto-Approval

```python
from typing import Callable
from udspy import ConfirmationRequired, ResumeState, ReAct

def create_smart_handler(
    auto_approve: Callable[[ConfirmationRequired], bool]
) -> Callable[[ConfirmationRequired], str]:
    """
    Create a handler that auto-approves based on custom logic.

    Args:
        auto_approve: Function that returns True if safe to auto-approve

    Returns:
        Handler function for confirmations
    """
    def handler(e: ConfirmationRequired) -> str:
        if auto_approve(e):
            print(f"✓ Auto-approved: {e.tool_call.name if e.tool_call else e.question}")
            return "yes"
        else:
            print(f"⚠️  Manual review: {e.question}")
            if e.tool_call:
                print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")
            return input("   Approve? (yes/no): ")

    return handler

# Define approval rules
def is_safe_operation(e: ConfirmationRequired) -> bool:
    """Check if operation is safe to auto-approve."""
    if not e.tool_call:
        return False

    # Auto-approve reads
    if e.tool_call.name in ["search", "read_file", "list_files"]:
        return True

    # Auto-approve writes to /tmp
    if e.tool_call.name in ["write_file", "delete_file"]:
        path = e.tool_call.args.get("path", "")
        if path.startswith("/tmp/"):
            return True

    return False

# Use the smart handler
handler = create_smart_handler(is_safe_operation)

result = run_with_confirmations(
    agent,
    question="Search logs, delete /tmp/cache, delete /prod/database",
    on_confirmation=handler
)
```

### Example 8: Timeout-based Confirmation

```python
import asyncio
from udspy import ConfirmationRequired, ResumeState, ReAct

async def get_user_input_with_timeout(
    question: str,
    timeout_seconds: int = 30,
    default_response: str = "no"
) -> str:
    """Get user input with timeout, returning default if no response."""
    print(f"\n⚠️  {question}")
    print(f"   (Timeout in {timeout_seconds}s, default: {default_response})")

    try:
        # Run input in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: input("   Response: ")),
            timeout=timeout_seconds
        )
        return response.strip()

    except asyncio.TimeoutError:
        print(f"   ⏱️  Timeout - using default: {default_response}")
        return default_response

async def run_with_timeout(agent, question):
    """Run agent with timeout-based confirmations."""
    resume_state = None

    while True:
        try:
            result = await agent.aforward(
                question=question,
                resume_state=resume_state
            )
            return result

        except ConfirmationRequired as e:
            # Timeout after 30 seconds, default to "no" (safe)
            user_response = await get_user_input_with_timeout(
                e.question,
                timeout_seconds=30,
                default_response="no"
            )
            resume_state = ResumeState(e, response)
```

## Testing Confirmations

### Example 9: Testing with Mock Confirmations

```python
from udspy import ConfirmationRequired, ResumeState, ReAct, respond_to_confirmation

async def test_agent_with_confirmation():
    """Test agent behavior with confirmations."""
    agent = ReAct("question -> answer", tools=[delete_file])

    # First call - should raise confirmation
    try:
        result = await agent.aforward(question="Delete /tmp/test.txt")
        assert False, "Should have raised ConfirmationRequired"
    except ConfirmationRequired as e:
        assert "delete" in e.question.lower()
        assert e.tool_call.name == "delete_file"

        # Programmatically approve
        respond_to_confirmation(e.confirmation_id, approved=True)

    # Resume - should complete
    result = await agent.aforward(
        question="Delete /tmp/test.txt",
        resume_state=e,
        user_response="yes"
    )

    assert "deleted" in result.answer.lower()
```

## See Also

- [Confirmation Architecture](../architecture/confirmation.md) - Design and implementation details
- [Confirmation API](../api/confirmation.md) - API reference
- [ReAct Examples](react.md) - ReAct-specific confirmation examples
- [Tool API](../api/tool.md) - Creating confirmable tools
