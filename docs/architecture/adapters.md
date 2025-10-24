# Adapters

Adapters handle the translation between Signatures and LLM-specific message formats.

## Overview

The `ChatAdapter` is responsible for:

1. Converting signatures into system prompts
2. Formatting inputs into user messages
3. Parsing LLM completions into structured outputs
4. Converting Pydantic models to tool schemas

## Usage

```python
from udspy import ChatAdapter

adapter = ChatAdapter()
```

Adapters are typically used internally by modules, but can be used directly:

```python
# Format instructions
instructions = adapter.format_instructions(signature)

# Format inputs
formatted = adapter.format_inputs(signature, {"question": "What is AI?"})

# Parse outputs
outputs = adapter.parse_outputs(signature, completion_text)
```

## Custom Adapters

You can create custom adapters by subclassing `ChatAdapter`:

```python
class CustomAdapter(ChatAdapter):
    def format_instructions(self, signature):
        # Custom instruction formatting
        return super().format_instructions(signature) + "\nBe creative!"
```

See [API: Adapters](../api/adapter.md) for detailed documentation.
