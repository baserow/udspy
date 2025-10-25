# API Reference: Signatures

Signatures define the inputs and outputs for modules. They provide type safety and clear contracts for LLM interactions.

## Creating Signatures

### Class-based Signatures

The traditional way to define signatures using Python classes:

```python
from udspy import Signature, InputField, OutputField

class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()
```

### String Signatures (DSPy-style)

For quick prototyping, use the string format `"inputs -> outputs"`:

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

- Inputs and outputs are comma-separated
- Whitespace is trimmed automatically
- All fields default to `str` type
- Optional second argument for instructions

### Direct Module Usage

All modules automatically recognize string signatures:

```python
from udspy import Predict, ChainOfThought

# These are equivalent
predictor1 = Predict("question -> answer")
predictor2 = Predict(Signature.from_string("question -> answer"))
```

## When to Use Each Format

**Use string signatures** (`from_string`) when:
- Prototyping quickly
- All fields are strings
- You don't need field descriptions
- The signature is simple

**Use class-based signatures** when:
- You need custom types (int, list, custom Pydantic models)
- You want field descriptions for better LLM guidance
- The signature is complex
- You want IDE autocomplete and type checking

## Examples

### Basic String Signature

```python
from udspy import Predict

predictor = Predict("question -> answer")
result = predictor(question="What is Python?")
print(result.answer)
```

### Multiple Fields

```python
from udspy import ChainOfThought

cot = ChainOfThought("context, question -> summary, answer")
result = cot(
    context="Python is a programming language",
    question="What is Python?"
)
print(result.reasoning)
print(result.summary)
print(result.answer)
```

### With Instructions

```python
QA = Signature.from_string(
    "question -> answer",
    "Answer questions concisely and accurately"
)
predictor = Predict(QA)
```

### Comparison

```python
# String format - quick and simple
QA_String = Signature.from_string("question -> answer")

# Class format - more control
class QA_Class(Signature):
    """Answer questions."""
    question: str = InputField(description="Question to answer")
    answer: str = OutputField(description="Concise answer")
```

## API Reference

::: udspy.signature
