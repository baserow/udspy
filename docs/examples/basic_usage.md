# Basic Usage

This guide covers the fundamentals of using udspy.

## Setup

First, configure the OpenAI client:

```python
import udspy

udspy.settings.configure(api_key="sk-...")
```

Or use environment variables:

```python
import os
import udspy

udspy.settings.configure(api_key=os.getenv("OPENAI_API_KEY"))
```

## Simple Question Answering

```python
from udspy import Signature, InputField, OutputField, Predict

class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)
result = predictor(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

## With Reasoning

```python
class ReasonedQA(Signature):
    """Answer questions with reasoning."""
    question: str = InputField()
    reasoning: str = OutputField(description="Step-by-step reasoning")
    answer: str = OutputField(description="Final answer")

predictor = Predict(ReasonedQA)
result = predictor(question="What is 15 * 23?")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
```

## Custom Model Parameters

```python
predictor = Predict(
    signature=QA,
    model="gpt-4",
    temperature=0.7,
    max_tokens=100,
)
```

## Global Defaults

```python
udspy.settings.configure(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7,
)

# All predictors use these defaults unless overridden
predictor = Predict(QA)
```

## Error Handling

```python
from pydantic import ValidationError

try:
    result = predictor(question="What is AI?")
except ValidationError as e:
    print(f"Output validation failed: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Testing

Mock the OpenAI client for testing:

```python
from unittest.mock import MagicMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage, Choice

def test_qa():
    # Mock response
    mock_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\nParis",
                ),
                finish_reason="stop",
            )
        ],
    )

    # Configure mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    udspy.settings.configure(client=mock_client)

    # Test
    predictor = Predict(QA)
    result = predictor(question="What is the capital of France?")
    assert result.answer == "Paris"
```

## Next Steps

- Learn about [Streaming](streaming.md)
- Explore [Tool Calling](tool_calling.md)
- See [Advanced Examples](advanced.md)
