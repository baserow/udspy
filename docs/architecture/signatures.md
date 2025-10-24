# Signatures

Signatures define the input/output contract for LLM tasks using Pydantic models.

## Basic Signature

```python
from udspy import Signature, InputField, OutputField

class QA(Signature):
    """Answer questions concisely and accurately."""
    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")
```

## Components

### Docstring

The class docstring becomes the task instruction in the system prompt:

```python
class Summarize(Signature):
    """Summarize the given text in 2-3 sentences."""
    text: str = InputField()
    summary: str = OutputField()
```

### InputField

Marks a field as an input:

```python
question: str = InputField(
    description="Question to answer",  # Used in prompt
    default="",  # Optional default value
)
```

### OutputField

Marks a field as an output:

```python
answer: str = OutputField(
    description="Concise answer",
)
```

## Field Types

Signatures support various field types:

### Primitives

```python
class Example(Signature):
    """Example signature."""
    text: str = InputField()
    count: int = InputField()
    score: float = InputField()
    enabled: bool = InputField()
```

### Collections

```python
class Example(Signature):
    """Example signature."""
    tags: list[str] = InputField()
    metadata: dict[str, Any] = InputField()
```

### Pydantic Models

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

class Example(Signature):
    """Example signature."""
    person: Person = InputField()
    related: list[Person] = OutputField()
```

## Dynamic Signatures

Create signatures programmatically:

```python
from udspy import make_signature

QA = make_signature(
    input_fields={"question": str},
    output_fields={"answer": str},
    instructions="Answer questions concisely",
)
```

## Validation

Signatures use Pydantic for validation:

```python
class Sentiment(Signature):
    """Analyze sentiment."""
    text: str = InputField()
    sentiment: Literal["positive", "negative", "neutral"] = OutputField()

# Output will be validated to match literal values
```

## Multi-Output Signatures

Signatures can have multiple outputs:

```python
class ReasonedQA(Signature):
    """Answer with step-by-step reasoning."""
    question: str = InputField()
    reasoning: str = OutputField(description="Reasoning process")
    answer: str = OutputField(description="Final answer")
```

## Best Practices

### 1. Clear Descriptions

```python
# Good
question: str = InputField(description="User's question about the product")

# Bad
question: str = InputField()
```

### 2. Specific Instructions

```python
# Good
class Summarize(Signature):
    """Summarize in exactly 3 bullet points, each under 20 words."""

# Bad
class Summarize(Signature):
    """Summarize."""
```

### 3. Structured Outputs

```python
# Good - use Pydantic models for complex outputs
class Analysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

class Analyze(Signature):
    """Analyze text."""
    text: str = InputField()
    analysis: Analysis = OutputField()

# Bad - use many separate fields
class Analyze(Signature):
    """Analyze text."""
    text: str = InputField()
    sentiment: str = OutputField()
    confidence: float = OutputField()
    keywords: list[str] = OutputField()
```

## API Reference

See [API: Signatures](../api/signature.md) for detailed API documentation.
