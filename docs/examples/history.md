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
- **Manages the system prompt** - Always keeps it at position 0, derived from your signature
- Adds user messages when you call the predictor
- Adds assistant responses after generation
- Maintains tool calls and results (when using tool calling)
- Preserves conversation context across turns

The system prompt is automatically set based on your signature, so you typically only need to track user/assistant messages. This makes managing conversation history much simpler!

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

# Set system message (always at position 0, replaces existing)
history.set_system_message("You are a helpful tutor")

# Add system message (appends to end - use set_system_message() instead)
history.add_system_message("You are a helpful tutor")  # Not recommended

# Add tool result
history.add_tool_result(tool_call_id="call_123", content="Result: 42")

# Add generic message
history.add_message("user", "Custom message")
```

**Note**: Use `set_system_message()` instead of `add_system_message()` to ensure the system prompt is always at position 0. When using `Predict`, the system prompt is automatically managed based on your signature, so you rarely need to set it manually.

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

## Automatic System Prompt Management

One of History's key features is automatic system prompt management. When you pass a History to Predict, the system prompt is automatically:

1. **Set from your signature** - The prompt is derived from your signature's docstring and fields
2. **Placed at position 0** - Always the first message in the conversation
3. **Updated on each call** - Keeps in sync with your signature if you change predictors

This means you can focus on tracking the actual conversation (user/assistant messages) and let udspy handle the system prompt:

```python
from udspy import History, Predict, Signature, InputField, OutputField

class QA(Signature):
    """Answer questions about programming."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)

# Start with an empty history
history = History()
print(f"Messages: {len(history)}")  # 0

# First call - system prompt automatically added
result = predictor(question="What is Python?", history=history)
print(f"Messages: {len(history)}")  # 2 (system + user)
print(f"First message role: {history.messages[0]['role']}")  # "system"

# Second call - system prompt stays at position 0
result = predictor(question="What about JavaScript?", history=history)
print(f"Messages: {len(history)}")  # 4 (system + user + assistant + user)
print(f"First message role: {history.messages[0]['role']}")  # Still "system"
```

### Pre-populating with User Messages Only

You can create a history with just conversation context, and the system prompt will be automatically prepended:

```python
# Load previous conversation from database (user/assistant only)
history = History()
history.add_user_message("Tell me about Python")
history.add_assistant_message("Python is a programming language...")
history.add_user_message("Is it beginner friendly?")
history.add_assistant_message("Yes! Python is great for beginners...")

print(f"First message role: {history.messages[0]['role']}")  # "user"

# Pass to Predict - system prompt prepended automatically
result = predictor(question="What about advanced features?", history=history)

print(f"First message role: {history.messages[0]['role']}")  # Now "system"!
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

# Pre-populate with previous conversation (user/assistant only)
# System prompt will be automatically added by Predict
history.add_user_message("I'm learning Python")
history.add_assistant_message("Great! I'm here to help.")

# System prompt is automatically prepended at position 0
result = predictor(question="How do I use list comprehensions?", history=history)

# history.messages[0] is now the system prompt from the signature
# history.messages[1] is "I'm learning Python"
# history.messages[2] is "Great! I'm here to help."
```

**Tip**: You only need to track user/assistant messages. The system prompt is automatically managed based on your signature.

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
