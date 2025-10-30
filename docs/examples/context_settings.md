# Context-Specific Settings

Learn how to use different API keys, models, and settings in different contexts.

## Overview

The `settings.context()` context manager allows you to temporarily override global settings for specific operations. This is useful for:

- Multi-tenant applications with different API keys per user
- Testing with different models
- Varying temperature or other parameters per request
- Isolating settings in async operations

The context manager is **thread-safe** using Python's `contextvars`, making it safe for concurrent operations.

## Basic Usage

### Override Model

```python
import udspy
from udspy import LM

# Configure global settings
global_lm = LM(model="gpt-4o-mini", api_key="sk-global")
udspy.settings.configure(lm=global_lm)

# Temporarily use a different model
gpt4_lm = LM(model="gpt-4", api_key="sk-global")
with udspy.settings.context(lm=gpt4_lm):
    predictor = Predict(QA)
    result = predictor(question="What is AI?")
    # Uses gpt-4

# Back to global settings (gpt-4o-mini)
result = predictor(question="What is ML?")
```

### Override API Key

```python
# Use a different API key for specific requests
user_lm = LM(model="gpt-4o-mini", api_key="sk-user-specific")
with udspy.settings.context(lm=user_lm):
    result = predictor(question="User-specific query")
    # Uses the user-specific API key
```

### Override Multiple Settings

```python
gpt4_lm = LM(model="gpt-4", api_key="sk-...")
with udspy.settings.context(
    lm=gpt4_lm,
    temperature=0.0,
    max_tokens=500
):
    result = predictor(question="Deterministic response needed")
```

## Multi-Tenant Applications

Handle different users with different API keys:

```python
from udspy import LM

def handle_user_request(user_id: str, question: str):
    """Handle a request from a specific user."""
    # Get user-specific API key from database
    user_api_key = get_user_api_key(user_id)

    # Use user's API key for this request
    user_lm = LM(model="gpt-4o-mini", api_key=user_api_key)
    with udspy.settings.context(lm=user_lm):
        predictor = Predict(QA)
        result = predictor(question=question)

    return result.answer

# Each user's request uses their own API key
answer1 = handle_user_request("user1", "What is Python?")
answer2 = handle_user_request("user2", "What is Rust?")
```

## Nested Contexts

Contexts can be nested, with inner contexts overriding outer ones:

```python
from udspy import LM

global_lm = LM(model="gpt-4o-mini", api_key="sk-...")
udspy.settings.configure(lm=global_lm, temperature=0.7)

gpt4_lm = LM(model="gpt-4", api_key="sk-...")
with udspy.settings.context(lm=gpt4_lm, temperature=0.5):
    # Uses gpt-4, temp=0.5

    with udspy.settings.context(temperature=0.0):
        # Uses gpt-4 (inherited), temp=0.0 (overridden)
        pass

    # Back to gpt-4, temp=0.5

# Back to gpt-4o-mini, temp=0.7
```

## Async Support

Context managers work seamlessly with async code:

```python
import asyncio
from udspy import LM

async def generate_response(question: str, user_api_key: str):
    user_lm = LM(model="gpt-4o-mini", api_key=user_api_key)
    with udspy.settings.context(lm=user_lm):
        predictor = StreamingPredict(QA)
        async for chunk in predictor.stream(question=question):
            yield chunk

# Handle multiple users concurrently
async def main():
    tasks = [
        generate_response("Question 1", "sk-user1"),
        generate_response("Question 2", "sk-user2"),
    ]
    await asyncio.gather(*tasks)
```

## Testing

Use contexts to isolate test settings:

```python
from udspy import LM

def test_with_specific_model():
    """Test behavior with a specific model."""
    test_lm = LM(model="gpt-4", api_key="sk-test")
    with udspy.settings.context(
        lm=test_lm,
        temperature=0.0,  # Deterministic for testing
    ):
        predictor = Predict(QA)
        result = predictor(question="2+2")
        assert "4" in result.answer
```

## Custom Endpoints

You can use custom endpoints with the LM factory:

```python
from udspy import LM

# Use custom endpoint
custom_lm = LM(
    model="custom-model",
    api_key="sk-custom",
    base_url="https://custom-endpoint.example.com/v1",
)

with udspy.settings.context(lm=custom_lm):
    # Uses custom endpoint
    result = predictor(question="...")
```

## Complete Example

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict, LM

# Global configuration
default_lm = LM(model="gpt-4o-mini", api_key="sk-default")
udspy.settings.configure(lm=default_lm, temperature=0.7)

class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)

# Scenario 1: Default settings
result = predictor(question="What is AI?")

# Scenario 2: High-quality request (use GPT-4)
gpt4_lm = LM(model="gpt-4", api_key="sk-default")
with udspy.settings.context(lm=gpt4_lm):
    result = predictor(question="Explain quantum computing")

# Scenario 3: Deterministic response
with udspy.settings.context(temperature=0.0):
    result = predictor(question="What is 2+2?")

# Scenario 4: User-specific API key
user_lm = LM(model="gpt-4o-mini", api_key=user.api_key)
with udspy.settings.context(lm=user_lm):
    result = predictor(question=user.question)
```

See the [full example](https://github.com/silvestrid/udspy/blob/main/examples/context.py) in the repository.
