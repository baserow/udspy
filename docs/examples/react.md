# ReAct (Reasoning and Acting)

ReAct is a powerful pattern for building LLM agents that can reason through multi-step problems and use tools to accomplish tasks. The name comes from combining **Rea**soning and **Act**ing.

## Overview

The ReAct module enables you to build agents that:

- **Reason iteratively**: Think through problems step-by-step
- **Use multiple tools**: Call different tools to gather information or perform actions
- **Handle ambiguity**: Ask users for clarification when needed
- **Require confirmation**: Request user approval for destructive operations
- **Save and restore state**: Pause execution for user input and resume seamlessly

## How ReAct Works

ReAct follows a **thought → action → observation** loop:

1. **Thought**: The agent reasons about what to do next
2. **Action**: The agent selects a tool and specifies arguments
3. **Observation**: The tool returns a result
4. **Repeat**: Continue until the task is complete

All reasoning steps are tracked in a **trajectory** that provides context for subsequent decisions.

## Basic Usage

```python
from pydantic import Field
from udspy import InputField, OutputField, ReAct, Signature, tool

# Define tools
@tool(name="search", description="Search for information")
def search(query: str = Field(description="Search query")) -> str:
    # Call search API
    return f"Search results for: {query}"

@tool(name="calculator", description="Perform calculations")
def calculator(expression: str = Field(description="Math expression")) -> str:
    return str(eval(expression))

# Define task signature
class ResearchTask(Signature):
    """Research a topic and provide a comprehensive answer."""
    question: str = InputField()
    answer: str = OutputField()

# Create ReAct agent
agent = ReAct(
    ResearchTask,
    tools=[search, calculator],
    max_iters=10
)

# Execute
result = agent(question="What is Python and how many letters are in 'Python'?")
print(result.answer)
# The agent will:
# 1. Search for "Python"
# 2. Calculate len("Python") = 6
# 3. Synthesize an answer combining both results
```

## String Signatures

For quick prototyping, you can use string signatures:

```python
agent = ReAct(
    "task -> result",  # Simple format: inputs -> outputs
    tools=[search, calculator]
)

result = agent(task="Find information about React")
print(result.result)
```

## User Clarification with `ask_to_user`

When the user's request is ambiguous, the agent can ask for clarification:

```python
from udspy import ConfirmationRequired, ResumeState

agent = ReAct(
    ResearchTask,
    tools=[search],
    enable_ask_to_user=True  # Enable clarification requests
)

try:
    result = agent(question="Tell me about it")
except ConfirmationRequired as e:
    # Agent needs clarification
    print(f"Agent asks: {e.question}")
    # "What topic would you like to know about?"

    # User provides clarification
    response = "The Python programming language"

    # Resume execution
    result = agent.resume(user_response, e)
    print(result.answer)
```

### Configuring `ask_to_user`

The `ask_to_user` tool is enabled by default and can be called by the agent whenever clarification is needed:

```python
agent = ReAct(
    ResearchTask,
    tools=[search],
    enable_ask_to_user=True,  # Default: enabled
)
```

To disable `ask_to_user` entirely:

```python
agent = ReAct(
    ResearchTask,
    tools=[search],
    enable_ask_to_user=False  # No clarification requests
)
```

## Tool Confirmation

For destructive or sensitive operations, you can require user confirmation:

```python
@tool(
    name="delete_file",
    description="Delete a file",
    require_confirmation=True  # Require confirmation
)
def delete_file(path: str = Field(description="File path")) -> str:
    os.remove(path)
    return f"Deleted {path}"

agent = ReAct(
    ResearchTask,
    tools=[delete_file]
)

try:
    result = agent(question="Delete /tmp/old_data.txt")
except ConfirmationRequired as e:
    # Agent asks for confirmation
    print(f"Confirm: {e.question}")
    # "Confirm execution of delete_file with args: {'path': '/tmp/old_data.txt'}? (yes/no)"

    # Check tool call info
    if e.tool_call:
        print(f"Tool: {e.tool_call.name}")
        print(f"Args: {e.tool_call.args}")

    # User confirms
    result = agent.resume("yes", e)
```

## Accessing the Trajectory

The trajectory contains all reasoning steps and tool calls:

```python
result = agent(question="What is 2 + 2?")

# Access trajectory
for i in range(10):  # Max iterations
    observation_key = f"observation_{i}"
    if observation_key not in result.trajectory:
        break

    print(f"Step {i + 1}:")
    print(f"  Reasoning: {result.trajectory.get(f'reasoning_{i}', '')}")
    print(f"  Tool: {result.trajectory[f'tool_name_{i}']}")
    print(f"  Args: {result.trajectory[f'tool_args_{i}']}")
    print(f"  Observation: {result.trajectory[observation_key]}")
```

Example trajectory:
```
Step 1:
  Reasoning: I need to calculate 2 + 2
  Tool: calculator
  Args: {'expression': '2 + 2'}
  Observation: 4

Step 2:
  Reasoning: I have the answer
  Tool: finish
  Args: {}
  Observation: Task completed
```

## Configuration Options

```python
agent = ReAct(
    signature=ResearchTask,       # Task signature
    tools=[search, calculator],   # Available tools
    max_iters=10,                 # Maximum reasoning steps (default: 10)
    enable_ask_to_user=True       # Enable user clarification (default: True)
)
```

### Parameters

- **`signature`**: Signature class or string format (`"input -> output"`)
- **`tools`**: List of tool functions (decorated with `@tool`) or `Tool` objects
- **`max_iters`**: Maximum number of reasoning iterations before stopping
- **`enable_ask_to_user`**: Whether to enable the `ask_to_user` tool

## Async Support

ReAct fully supports async execution:

```python
import asyncio

async def main():
    agent = ReAct(ResearchTask, tools=[search])

    # Async forward
    result = await agent.aforward(question="What is Python?")
    print(result.answer)

    # Or use the async resume method
    try:
        result = await agent.aforward(question="Tell me about it")
    except ConfirmationRequired as e:
        result = await agent.aresume("Python", e)

asyncio.run(main())
```

## Built-in Tools

Every ReAct agent automatically includes these tools:

### `finish`

Signals that the agent has collected enough information to answer:

```python
# Agent internally calls:
# Tool: finish
# Args: {}
```

This is automatically selected by the LLM when it has sufficient information.

### `ask_to_user` (if enabled)

Requests clarification from the user:

```python
# Agent internally calls:
# Tool: ask_to_user
# Args: {"question": "What topic would you like to know about?"}
```

## Advanced Patterns

### Multi-Tool Research

```python
@tool(name="search_papers", description="Search academic papers")
def search_papers(query: str = Field(...)) -> str:
    return f"Papers about: {query}"

@tool(name="summarize", description="Summarize text")
def summarize(text: str = Field(...)) -> str:
    return f"Summary of: {text[:100]}..."

class DeepResearch(Signature):
    """Conduct deep research on a scientific topic."""
    topic: str = InputField()
    summary: str = OutputField()

agent = ReAct(
    DeepResearch,
    tools=[search_papers, summarize],
    max_iters=15  # More steps for complex research
)

result = agent(topic="quantum computing")
```

### Error Recovery

The agent automatically handles tool errors and can recover:

```python
@tool(name="api_call", description="Call external API")
def api_call(endpoint: str = Field(...)) -> str:
    try:
        # Simulated API call that might fail
        if endpoint == "invalid":
            raise ValueError("Invalid endpoint")
        return "API response"
    except Exception as e:
        # Error is returned as observation
        raise

# Agent will see error in observation and can:
# 1. Try a different tool
# 2. Retry with different args
# 3. Ask user for help (using ask_to_user tool)
```

### State Management

Save and restore execution state:

```python
try:
    result = agent(question="Delete important files")
except ConfirmationRequired as e:
    # Save state
    saved_state = e
    saved_question = e.question
    # Access ReAct-specific state from context
    saved_trajectory = e.context["trajectory"]
    saved_iteration = e.context["iteration"]
    saved_input_args = e.context["input_args"]

    # Later, restore and continue
    response = input(f"{saved_question} ")
    result = agent.resume(user_response, saved_state)
```

## DSPy Compatibility

The `Tool` class includes DSPy-compatible aliases:

```python
from udspy import Tool

@tool(name="search", description="Search tool")
def search(query: str = Field(...)) -> str:
    return "results"

# DSPy-style access
print(search.desc)   # Same as search.description
print(search.args)   # Dict of argument specs
```

## Best Practices

1. **Provide clear tool descriptions**: The LLM uses descriptions to select tools
2. **Use Field() for parameters**: Provide descriptions for all tool parameters
3. **Limit max_iters**: Prevent infinite loops with reasonable iteration limits
4. **Enable confirmation for destructive ops**: Use `require_confirmation=True`
5. **Handle ConfirmationRequired**: Always catch and handle clarification requests
6. **Use specific signatures**: Clear input/output fields help the agent understand the task
7. **Test with mock tools**: Use simple mock tools to validate agent logic

## Common Patterns

### Research and Summarize

```python
agent = ReAct(
    "query -> summary",
    tools=[search, summarize]
)
result = agent(query="Latest AI developments")
```

### Data Analysis

```python
@tool(name="load_data", description="Load dataset")
def load_data(path: str = Field(...)) -> str:
    return "data loaded"

@tool(name="analyze", description="Analyze data")
def analyze(metric: str = Field(...)) -> str:
    return "analysis results"

agent = ReAct(
    "dataset, question -> insights",
    tools=[load_data, analyze]
)
```

### Multi-Step Workflows

```python
@tool(name="step1", description="First step")
def step1() -> str: return "step1 done"

@tool(name="step2", description="Second step")
def step2(input: str = Field(...)) -> str: return "step2 done"

@tool(name="step3", description="Third step")
def step3(input: str = Field(...)) -> str: return "step3 done"

agent = ReAct(
    "task -> result",
    tools=[step1, step2, step3],
    max_iters=20
)
```

## Troubleshooting

### Agent doesn't finish

Increase `max_iters` or simplify the task:

```python
agent = ReAct(signature, tools=tools, max_iters=20)
```

### Too many tool calls

Reduce `max_iters` or improve tool descriptions:

```python
@tool(
    name="search",
    description="Search ONLY when you need external information. Use for factual queries."
)
```

### Agent asks for clarification too often

Disable or restrict `ask_to_user`:

```python
agent = ReAct(
    signature,
    tools=tools,
    enable_ask_to_user=False  # Disable entirely
)
```

Or disable it completely if the agent should never ask for clarification:

```python
agent = ReAct(
    signature,
    tools=tools,
    enable_ask_to_user=False  # Disable user clarification
)
```

## Loop-Based Confirmation Handling

For applications requiring multiple confirmations or interactive workflows, use the loop-based pattern with `resume_state`:

### Basic Loop Pattern

```python
from udspy import ConfirmationRequired, ResumeState

resume_state = None

while True:
    try:
        result = agent(
            question="Your task here",
            resume_state=resume_state
        )
        print(f"Success: {result.answer}")
        break

    except ConfirmationRequired as e:
        print(f"Confirmation needed: {e.question}")
        response = input("Your response (yes/no): ")
        resume_state = ResumeState(e, response)
```

### Advantages of Loop Pattern

- **Multiple confirmations**: Handle as many confirmations as needed
- **Uniform handling**: Same code path for all confirmations
- **Stateless API calls**: Ideal for web APIs and distributed systems
- **Simpler control flow**: No need to track separate resume calls

### Comparison with `aresume()`

**Using `aresume()` (explicit):**
```python
try:
    result = await agent.aforward(question="Delete files")
except ConfirmationRequired as e:
    result = await agent.aresume("yes", e)
```

**Using loop pattern (uniform):**
```python
resume_state = None

while True:
    try:
        result = await agent.aforward(
            question="Delete files",
            resume_state=resume_state
        )
        break
    except ConfirmationRequired as e:
        response = input(f"{e.question} (yes/no): ")
        resume_state = ResumeState(e, response)
```

The loop pattern is especially useful when:
- You expect multiple confirmations
- Building interactive CLIs
- Implementing web API endpoints
- Want predictable, uniform handling

For comprehensive examples and patterns, see the [Confirmation Examples](confirmation.md) guide.

## See Also

- [Confirmation Examples](confirmation.md) - Detailed confirmation patterns
- [Confirmation Architecture](../architecture/confirmation.md) - Design and implementation
- [Tool Calling Guide](tool_calling.md) - Creating custom tools
- [Chain of Thought](chain_of_thought.md) - Simpler reasoning module
- [ReAct API Reference](../api/react.md) - Full API documentation
