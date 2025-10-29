# Conversation History

The `History` class manages conversation history for multi-turn interactions. When passed to `Predict`, it automatically maintains context across multiple calls.

## Basic Usage

```python
from udspy import History, Predict, Signature, InputField, OutputField

class QA(Signature):
    '''Answer questions.'''
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)
history = History()

# First turn
result = predictor(question="What is Python?", history=history)
print(result.answer)

# Second turn - context is maintained
result = predictor(question="What are its main features?", history=history)
print(result.answer)  # Assistant knows we're still talking about Python
```

## How It Works

`History` stores messages in OpenAI format and automatically:
- Adds user messages when you call the predictor
- Adds assistant responses after generation
- Maintains tool calls and results (when using tool calling)
- Preserves conversation context across turns

## API

### Creating History

```python
# Empty history
history = History()

# With initial messages
history = History(messages=[
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
])
```

### Adding Messages

```python
# Add user message
history.add_user_message("What is AI?")

# Add assistant message
history.add_assistant_message("AI stands for Artificial Intelligence...")

# Add system message
history.add_system_message("You are a helpful tutor")

# Add tool result
history.add_tool_result(tool_call_id="call_123", content="Result: 42")

# Add generic message
history.add_message("user", "Custom message")
```

### Managing History

```python
# Get number of messages
print(len(history))  # e.g., 5

# Clear all messages
history.clear()

# Copy history (for branching conversations)
branch = history.copy()

# Access messages directly
for msg in history.messages:
    print(f"{msg['role']}: {msg['content']}")

# String representation
print(history)  # Shows formatted conversation
```

## Use Cases

### Multi-Turn Conversations

```python
predictor = Predict(QA)
history = History()

# Each call maintains context
predictor(question="What is machine learning?", history=history)
predictor(question="How does it differ from traditional programming?", history=history)
predictor(question="Can you give me an example?", history=history)
```

### Pre-Populating Context

```python
history = History()

# Set up initial context
history.add_system_message("You are a Python expert. Keep answers concise.")
history.add_user_message("I'm learning Python")
history.add_assistant_message("Great! I'm here to help.")

# Now ask questions with this context
result = predictor(question="How do I use list comprehensions?", history=history)
```

### Branching Conversations

```python
main_history = History()

# Start main conversation
predictor(question="Tell me about programming languages", history=main_history)

# Branch 1: Explore Python
python_branch = main_history.copy()
predictor(question="Tell me more about Python", history=python_branch)

# Branch 2: Explore JavaScript
js_branch = main_history.copy()
predictor(question="Tell me more about JavaScript", history=js_branch)

# Each branch maintains independent context
```

### Conversation Reset

```python
history = History()

# First conversation
predictor(question="What is Python?", history=history)

# Reset for new topic
history.clear()

# New conversation with no context
predictor(question="What is JavaScript?", history=history)
```

### History with Tool Calling

```python
from udspy import tool
from pydantic import Field

@tool(name="Calculator", description="Perform calculations")
def calculator(operation: str = Field(...), a: float = Field(...), b: float = Field(...)) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

predictor = Predict(QA, tools=[calculator])
history = History()

# Tool calls are automatically recorded in history
result = predictor(question="What is 15 times 23?", history=history)
# History now contains: user message, assistant tool call, tool result, final assistant answer

# Next turn has full context including tool usage
result = predictor(question="Now add 100 to that", history=history)
```

## Best Practices

1. **One History per Conversation Thread**: Create a new `History` instance for each independent conversation
2. **Use `copy()` for Branching**: When you want to explore different paths from the same starting point
3. **Clear When Changing Topics**: Use `history.clear()` when starting a completely new conversation
4. **Pre-populate for Context**: Add system messages or previous conversation history to set context
5. **Inspect Messages**: Access `history.messages` directly when you need to debug or log conversations

## Async Support

History works seamlessly with all async patterns:

```python
# Async streaming
async for event in predictor.astream(question="...", history=history):
    if isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)

# Async non-streaming
result = await predictor.aforward(question="...", history=history)

# Sync (uses asyncio.run internally)
result = predictor(question="...", history=history)
```

## Examples

See [history.py](https://github.com/silvestrid/udspy/blob/main/examples/history.py) for complete working examples.
