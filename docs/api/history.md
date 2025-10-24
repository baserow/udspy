# History API Reference

The `History` class manages conversation history for multi-turn interactions.

## Class: History

```python
from udspy import History
```

### Constructor

```python
History(messages: list[dict[str, Any]] | None = None)
```

Create a new History instance.

**Parameters:**
- `messages` (optional): Initial list of messages in OpenAI format

**Example:**
```python
# Empty history
history = History()

# With initial messages
history = History(messages=[
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
])
```

### Attributes

#### messages

```python
history.messages: list[dict[str, Any]]
```

List of conversation messages in OpenAI format. Each message is a dictionary with at minimum:
- `role`: One of "system", "user", "assistant", or "tool"
- `content`: Message content string

Assistant messages with tool calls also include:
- `tool_calls`: List of tool call dictionaries

Tool messages also include:
- `tool_call_id`: ID of the tool call this result is for

**Example:**
```python
for msg in history.messages:
    print(f"{msg['role']}: {msg['content']}")
```

### Methods

#### add_message

```python
add_message(
    role: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None
) -> None
```

Add a message to the history.

**Parameters:**
- `role`: Message role ("system", "user", "assistant", "tool")
- `content`: Message content
- `tool_calls` (optional): Tool calls for assistant messages

**Example:**
```python
history.add_message("user", "What is AI?")
history.add_message("assistant", "AI is...", tool_calls=[...])
```

#### add_user_message

```python
add_user_message(content: str) -> None
```

Add a user message.

**Parameters:**
- `content`: User message content

**Example:**
```python
history.add_user_message("Tell me about Python")
```

#### add_assistant_message

```python
add_assistant_message(
    content: str,
    tool_calls: list[dict[str, Any]] | None = None
) -> None
```

Add an assistant message.

**Parameters:**
- `content`: Assistant message content
- `tool_calls` (optional): Tool calls made by assistant

**Example:**
```python
history.add_assistant_message("Python is a programming language...")
```

#### add_system_message

```python
add_system_message(content: str) -> None
```

Add a system message.

**Parameters:**
- `content`: System message content

**Example:**
```python
history.add_system_message("You are a helpful coding tutor")
```

#### add_tool_result

```python
add_tool_result(tool_call_id: str, content: str) -> None
```

Add a tool result message.

**Parameters:**
- `tool_call_id`: ID of the tool call this result is for
- `content`: Tool result content

**Example:**
```python
history.add_tool_result("call_123", "Result: 42")
```

#### clear

```python
clear() -> None
```

Clear all messages from history.

**Example:**
```python
history.clear()
print(len(history))  # 0
```

#### copy

```python
copy() -> History
```

Create a copy of this history.

**Returns:**
- New `History` instance with copied messages

**Example:**
```python
branch = history.copy()
# Modify branch without affecting original
branch.add_user_message("New question")
```

### Magic Methods

#### `__len__`

```python
len(history) -> int
```

Get number of messages in history.

**Example:**
```python
print(len(history))  # e.g., 5
```

#### `__repr__`

```python
repr(history) -> str
```

String representation showing number of messages.

**Example:**
```python
print(repr(history))  # "History(5 messages)"
```

#### `__str__`

```python
str(history) -> str
```

Human-readable formatted conversation history.

**Example:**
```python
print(history)
# History (3 messages):
#   1. [user] What is Python?
#   2. [assistant] Python is a programming language...
#   3. [user] What are its features?
```

## Usage with Predict

History integrates seamlessly with `Predict`:

```python
from udspy import Predict, History

predictor = Predict(QA)
history = History()

# History is automatically updated with each call
result = predictor(question="First question", history=history)
result = predictor(question="Follow-up question", history=history)
```

See [History Examples](../examples/history.md) for more usage patterns.
