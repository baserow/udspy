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

# Configure global settings
udspy.settings.configure(api_key="sk-global", model="gpt-4o-mini")

# Temporarily use a different model
with udspy.settings.context(model="gpt-4"):
    predictor = Predict(QA)
    result = predictor(question="What is AI?")
    # Uses gpt-4

# Back to global settings (gpt-4o-mini)
result = predictor(question="What is ML?")
```

### Override API Key

```python
# Use a different API key for specific requests
with udspy.settings.context(api_key="sk-user-specific"):
    result = predictor(question="User-specific query")
    # Uses the user-specific API key
```

### Override Multiple Settings

```python
with udspy.settings.context(
    model="gpt-4",
    temperature=0.0,
    max_tokens=500
):
    result = predictor(question="Deterministic response needed")
```

## Multi-Tenant Applications

Handle different users with different API keys:

```python
def handle_user_request(user_id: str, question: str):
    """Handle a request from a specific user."""
    # Get user-specific API key from database
    user_api_key = get_user_api_key(user_id)

    # Use user's API key for this request
    with udspy.settings.context(api_key=user_api_key):
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
udspy.settings.configure(model="gpt-4o-mini", temperature=0.7)

with udspy.settings.context(model="gpt-4", temperature=0.5):
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

async def generate_response(question: str, user_api_key: str):
    with udspy.settings.context(api_key=user_api_key):
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
def test_with_specific_model():
    """Test behavior with a specific model."""
    with udspy.settings.context(
        api_key="sk-test",
        model="gpt-4",
        temperature=0.0,  # Deterministic for testing
    ):
        predictor = Predict(QA)
        result = predictor(question="2+2")
        assert "4" in result.answer
```

## Custom Clients

You can also provide custom OpenAI clients:

```python
from openai import OpenAI, AsyncOpenAI

custom_client = OpenAI(
    api_key="sk-custom",
    base_url="https://custom-endpoint.example.com",
)

with udspy.settings.context(client=custom_client):
    # Uses custom client with custom endpoint
    result = predictor(question="...")
```

## Complete Example

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict

# Global configuration
udspy.settings.configure(
    api_key="sk-default",
    model="gpt-4o-mini",
    temperature=0.7,
)

class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

predictor = Predict(QA)

# Scenario 1: Default settings
result = predictor(question="What is AI?")

# Scenario 2: High-quality request (use GPT-4)
with udspy.settings.context(model="gpt-4"):
    result = predictor(question="Explain quantum computing")

# Scenario 3: Deterministic response
with udspy.settings.context(temperature=0.0):
    result = predictor(question="What is 2+2?")

# Scenario 4: User-specific API key
with udspy.settings.context(api_key=user.api_key):
    result = predictor(question=user.question)
```

See the [full example](../../examples/context_example.py) in the repository.
