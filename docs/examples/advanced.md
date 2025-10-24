# Advanced Examples

Advanced patterns and techniques.

## Module Composition

Build complex modules from simpler ones:

```python
from udspy import Module, Predict, Prediction

class ChainOfThought(Module):
    """Answer questions with explicit reasoning."""

    def __init__(self, signature):
        # Create intermediate signature for reasoning
        self.reason = Predict(make_signature(
            signature.get_input_fields(),
            {"reasoning": str},
            "Think step-by-step about this problem",
        ))

        # Final answer with reasoning context
        self.answer = Predict(signature)

    def forward(self, **inputs):
        # Generate reasoning
        thought = self.reason(**inputs)

        # Generate answer (could inject reasoning into prompt)
        result = self.answer(**inputs)
        result["reasoning"] = thought.reasoning

        return Prediction(**result)
```

## Retry Logic

Implement retry with validation:

```python
from pydantic import ValidationError

class ValidatedPredict(Module):
    def __init__(self, signature, max_retries=3):
        self.predictor = Predict(signature)
        self.signature = signature
        self.max_retries = max_retries

    def forward(self, **inputs):
        for attempt in range(self.max_retries):
            try:
                result = self.predictor(**inputs)
                # Validate result matches expected schema
                return result
            except ValidationError as e:
                if attempt == self.max_retries - 1:
                    raise
                # Could inject error message to help LLM correct
                continue
```

## Prompt Caching

Cache prompts for repeated queries:

```python
from functools import lru_cache

class CachedPredict(Module):
    def __init__(self, signature):
        self.predictor = Predict(signature)
        self._cached_predict = lru_cache(maxsize=128)(self._predict)

    def _predict(self, **inputs):
        # Convert inputs to hashable tuple
        key = tuple(sorted(inputs.items()))
        return self.predictor(**dict(key))

    def forward(self, **inputs):
        return self._cached_predict(**inputs)
```

## Custom Adapters

Create custom formatting:

```python
from udspy import ChatAdapter

class VerboseAdapter(ChatAdapter):
    def format_instructions(self, signature):
        base = super().format_instructions(signature)
        return f"{base}\n\nIMPORTANT: Be extremely detailed in your response."

    def format_inputs(self, signature, inputs):
        base = super().format_inputs(signature, inputs)
        return f"Input data:\n{base}\n\nAnalyze thoroughly:"

predictor = Predict(signature, adapter=VerboseAdapter())
```

## Testing Helpers

Utilities for testing:

```python
from unittest.mock import MagicMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage, Choice

def mock_openai_response(content: str) -> ChatCompletion:
    """Create a mock OpenAI response."""
    return ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
                finish_reason="stop",
            )
        ],
    )

def test_with_mock():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response(
        "[[ ## answer ## ]]\nTest answer"
    )

    udspy.settings.configure(client=mock_client)

    predictor = Predict(QA)
    result = predictor(question="Test?")
    assert result.answer == "Test answer"
```
