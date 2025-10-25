# Signatures

Signatures define the input/output contract for LLM tasks using Pydantic models.

## Creating Signatures

There are three ways to create signatures in udspy:

### 1. String Signatures (Quick & Simple)

For rapid prototyping, use the DSPy-style string format:

```python
from udspy import Signature

# Simple signature
QA = Signature.from_string("question -> answer")

# Multiple inputs and outputs
Analyze = Signature.from_string(
    "context, question -> summary, answer",
    "Analyze text and answer questions"
)
```

**Format**: `"input1, input2 -> output1, output2"`

- All fields default to `str` type
- Optional second argument for instructions
- Great for quick prototyping

### 2. Class-based Signatures (Full Control)

For production code, use class-based signatures:

```python
from udspy import Signature, InputField, OutputField

class QA(Signature):
    """Answer questions concisely and accurately."""
    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")
```

**Benefits**:
- Custom field types
- Field descriptions for better prompts
- IDE autocomplete and type checking
- Better for complex signatures

### 3. Dynamic Signatures (Programmatic)

For runtime signature creation:

```python
from udspy import make_signature

QA = make_signature(
    input_fields={"question": str},
    output_fields={"answer": str},
    instructions="Answer questions concisely",
)
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

## Choosing the Right Approach

### String Signatures

**Use when**:
- Prototyping quickly
- All fields are strings
- Signature is simple
- You don't need field descriptions

```python
# Perfect for quick tests
predictor = Predict("question -> answer")
```

### Class-based Signatures

**Use when**:
- Building production code
- You need custom types
- You want field descriptions
- Signature is complex

```python
class QA(Signature):
    """Answer questions."""
    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Concise answer")
```

### Dynamic Signatures

**Use when**:
- Creating signatures at runtime
- Building signature builders/factories
- Signature structure depends on config

```python
# Build signature based on config
fields = load_field_config()
MySignature = make_signature(fields["inputs"], fields["outputs"])
```

## Modules Accept All Formats

All modules automatically recognize string signatures:

```python
from udspy import Predict, ChainOfThought, ReAct

# All of these work:
predictor1 = Predict("question -> answer")
predictor2 = Predict(QA)  # Class-based
predictor3 = Predict(make_signature(...))  # Dynamic

cot = ChainOfThought("question -> answer")
agent = ReAct("question -> answer", tools=[...])
```

## API Reference

See [API: Signatures](../api/signature.md) for detailed API documentation including the full `Signature.from_string()` reference.
