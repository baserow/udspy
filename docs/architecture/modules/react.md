# ReAct Module

The `ReAct` (Reasoning and Acting) module implements an agent that iteratively reasons about tasks and uses tools to accomplish goals.

## Overview

ReAct combines:

- **Reasoning**: Step-by-step thinking about what to do next
- **Acting**: Calling tools to perform actions
- **Iteration**: Repeating until the task is complete

This creates an agent that can break down complex tasks, use available tools, and ask for help when needed.

## Basic Usage

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

agent = ReAct(QA, tools=[search])
result = agent(question="What is the weather in Tokyo?")

print(result.answer)
print(result.trajectory)  # Full reasoning history
```

## String Signatures

For quick prototyping:

```python
agent = ReAct("question -> answer", tools=[search])
result = agent(question="What is Python?")
```

## How ReAct Works

### Iteration Loop

1. **Reason**: Agent thinks about current situation
2. **Act**: Agent calls a tool (or finish)
3. **Observe**: Agent sees tool result
4. **Repeat**: Until agent calls `finish` tool or max iterations

### Built-in Tools

ReAct automatically provides:

- **finish**: Call when task is complete
- **ask_to_user**: Ask user for clarification (if enabled)

```python
# The agent automatically has these tools available:
# - finish(answer: str) - Complete the task
# - ask_to_user(question: str) - Ask user for help
```

### Trajectory

The trajectory records every step:

```python
result = agent(question="Calculate 15 * 23")

# Access trajectory
print(result.trajectory)
# {
#   "reasoning_0": "I need to calculate 15 * 23",
#   "tool_name_0": "calculator",
#   "tool_args_0": {"expression": "15 * 23"},
#   "observation_0": "345",
#   "reasoning_1": "I have the answer",
#   "tool_name_1": "finish",
#   ...
# }
```

## Configuration

### Maximum Iterations

```python
agent = ReAct(QA, tools=[search], max_iters=10)
result = agent(question="...", max_iters=5)  # Override per call
```

### Disable Ask-to-User

```python
agent = ReAct(QA, tools=[search], enable_ask_to_user=False)
```

## Human-in-the-Loop

ReAct supports tools with require_confirmation that require human confirmation:

```python
from udspy import ConfirmationRequired, tool

@tool(name="delete_file", require_confirmation=True)
def delete_file(path: str = Field(...)) -> str:
    return f"Deleted {path}"

agent = ReAct(QA, tools=[delete_file])

try:
    result = await agent.aforward(question="Delete /tmp/test.txt")
except ConfirmationRequired as e:
    print(f"Confirm: {e.question}")
    print(f"Tool: {e.tool_call.name}")
    print(f"Args: {e.tool_call.args}")

    # User approves
    result = await agent.aresume("yes", e)

    # Or user rejects
    result = await agent.aresume("no", e)

    # Or user modifies arguments
    result = await agent.aresume('{"path": "/tmp/other.txt"}', e)
```

### Resumption Flow

When a confirmation is requested:

1. Agent pauses and raises `ConfirmationRequired`
2. Exception contains saved state and pending tool call
3. User reviews and responds
4. Call `aresume(response, saved_state)` to continue

See [Confirmation API](../../api/confirmation.md) for details.

## Streaming

Stream the agent's reasoning in real-time:

```python
async for event in agent.aexecute(
    stream=True,
    question="What is quantum computing?"
):
    if isinstance(event, StreamChunk):
        if event.field == "reasoning":
            print(f"Thinking: {event.delta}", end="", flush=True)
    elif isinstance(event, Prediction):
        print(f"\n\nAnswer: {event.answer}")
```

See `examples/react_streaming.py` for a complete example.

## Architecture

### Internal Signatures

ReAct uses two internal signatures:

1. **react_signature**: For reasoning and tool selection
   - Inputs: Original inputs + trajectory
   - Outputs: reasoning
   - Tools: All provided tools + finish

2. **extract_signature**: For extracting final answer
   - Inputs: Original inputs + trajectory
   - Outputs: Original outputs
   - Uses ChainOfThought for extraction

### Modules

ReAct composes two modules:

- `react_module`: Predict with tools for reasoning/acting
- `extract_module`: ChainOfThought for final answer extraction

### Example Flow

```
User: "What is the capital of France?"

Iteration 0:
  Reasoning: "I need to search for France's capital"
  Tool: search
  Args: {"query": "capital of France"}
  Observation: "Paris is the capital of France"

Iteration 1:
  Reasoning: "I have the answer, I can finish"
  Tool: finish
  Args: {}
  Observation: "Task completed"

Extract:
  Reasoning: "Based on the search, Paris is the capital"
  Answer: "Paris"
```

## Advanced Usage

### Custom Tools

```python
from pydantic import Field

@tool(
    name="calculator",
    description="Evaluate mathematical expressions"
)
def calc(expression: str = Field(description="Math expression")) -> str:
    return str(eval(expression))

@tool(
    name="web_search",
    description="Search the web for information"
)
async def web_search(query: str = Field(...)) -> str:
    # Async tools are supported
    return await search_api(query)

agent = ReAct(QA, tools=[calc, web_search])
```

### Multiple Outputs

```python
class Research(Signature):
    """Research a topic thoroughly."""
    topic: str = InputField()
    summary: str = OutputField()
    sources: str = OutputField()
    confidence: str = OutputField()

agent = ReAct(Research, tools=[search])
result = agent(topic="Quantum Computing")

print(result.summary)
print(result.sources)
print(result.confidence)
```

### Tool Error Handling

Tools can raise exceptions - they're caught and added to observations:

```python
@tool(name="divide")
def divide(a: int = Field(...), b: int = Field(...)) -> str:
    return str(a / b)

agent = ReAct(QA, tools=[divide])
result = agent(question="What is 10 divided by 0?")

# Agent sees: "Error executing divide: division by zero"
# Agent can reason about the error and try alternative approaches
```

## Design Rationale

### Why Two Phases (React + Extract)?

1. **react_module**: Focuses on tool usage and reasoning
2. **extract_module**: Focuses on clean output formatting

This separation ensures:
- Tool-using prompts stay focused on actions
- Final outputs are well-formatted
- Trajectory doesn't pollute final answer

### Why ask_to_user Tool?

The built-in `ask_to_user` tool allows agents to:
- Request clarification when ambiguous
- Ask for additional information
- Interact naturally with users

It's implemented as a tool with require_confirmation, so users can provide responses that the agent incorporates into its reasoning.

### Why finish Tool?

The `finish` tool signals task completion:
- Explicit end condition (vs implicit max iterations)
- Agent decides when it has enough information
- More natural than counting iterations

## Common Patterns

### Research Agent

```python
@tool(name="search")
def search(query: str = Field(...)) -> str:
    return search_web(query)

@tool(name="summarize")
def summarize(text: str = Field(...)) -> str:
    return llm_summarize(text)

researcher = ReAct(
    "topic -> summary, sources",
    tools=[search, summarize]
)
result = researcher(topic="AI Safety")
```

### Task Automation

```python
@tool(name="read_file")
def read_file(path: str = Field(...)) -> str:
    return open(path).read()

@tool(name="write_file", require_confirmation=True)
def write_file(path: str = Field(...), content: str = Field(...)) -> str:
    with open(path, 'w') as f:
        f.write(content)
    return f"Wrote to {path}"

assistant = ReAct(
    "task -> result",
    tools=[read_file, write_file]
)
```

### Multi-tool Problem Solving

```python
@tool(name="calculator")
def calc(expr: str = Field(...)) -> str:
    return str(eval(expr))

@tool(name="unit_converter")
def convert(value: float = Field(...), from_unit: str = Field(...), to_unit: str = Field(...)) -> str:
    # Conversion logic
    return f"{result} {to_unit}"

solver = ReAct(
    "problem -> solution",
    tools=[calc, convert]
)
result = solver(problem="Convert 100 fahrenheit to celsius and add 10")
```

## Limitations

1. **Token Usage**: Each iteration adds to token count
2. **Latency**: Multiple LLM calls increase response time
3. **Reliability**: Agent may not always pick the right tool
4. **Max Iterations**: Tasks may not complete within iteration limit

## See Also

- [Base Module](base.md) - Module foundation
- [Predict Module](predict.md) - Core prediction
- [Tool API](../../api/tool.md) - Creating tools
- [Confirmation API](../../api/confirmation.md) - Human-in-the-loop
- [ADR-005: ReAct Module](../decisions.md#adr-005-react-agent-module)
- [ADR-004: Confirmation System](../decisions.md#adr-004-human-in-the-loop-with-confirmation-system)
