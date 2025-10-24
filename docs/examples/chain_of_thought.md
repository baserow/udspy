# Chain of Thought Examples

Chain of Thought (CoT) is a prompting technique that improves LLM reasoning by explicitly requesting step-by-step thinking before producing the final answer.

## Overview

The `ChainOfThought` module automatically adds a "reasoning" field to any signature, encouraging the LLM to show its work:

```python
from udspy import ChainOfThought, Signature, InputField, OutputField

class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

# Automatically adds reasoning step
cot = ChainOfThought(QA)
result = cot(question="What is 15 * 23?")

print(result.reasoning)  # Step-by-step calculation
print(result.answer)     # "345"
```

## Basic Usage

### Simple Question Answering

```python
import udspy

udspy.settings.configure(api_key="your-key")

class QA(Signature):
    """Answer questions clearly."""
    question: str = InputField()
    answer: str = OutputField()

predictor = ChainOfThought(QA)
result = predictor(question="What is the capital of France?")

print("Reasoning:", result.reasoning)
# "Let me recall the capital cities of European countries.
#  France is a major European nation, and its capital is Paris."

print("Answer:", result.answer)
# "Paris"
```

### Math Problems

Chain of Thought excels at mathematical reasoning:

```python
result = predictor(question="What is 17 * 24?")

print("Reasoning:", result.reasoning)
# "I'll break this down: 17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408"

print("Answer:", result.answer)
# "408"
```

## Advanced Usage

### Custom Reasoning Description

Customize how the reasoning field is described:

```python
cot = ChainOfThought(
    QA,
    reasoning_description="Detailed mathematical proof with all intermediate steps"
)

result = cot(question="Prove that the sum of angles in a triangle is 180 degrees")
```

### Multiple Output Fields

Chain of Thought works with signatures that have multiple outputs:

```python
class Analysis(Signature):
    """Analyze text comprehensively."""
    text: str = InputField()
    summary: str = OutputField(description="Brief summary")
    sentiment: str = OutputField(description="Sentiment analysis")
    keywords: list[str] = OutputField(description="Key terms")

analyzer = ChainOfThought(Analysis)
result = analyzer(text="Long article text here...")

# Access all outputs plus reasoning
print(result.reasoning)   # Analysis process
print(result.summary)     # Summary
print(result.sentiment)   # Sentiment
print(result.keywords)    # Keywords
```

### With Custom Model Parameters

```python
# Use with specific model and temperature
cot = ChainOfThought(
    QA,
    model="gpt-4",
    temperature=0.0,  # Deterministic for math
)

result = cot(question="What is the square root of 144?")
```

## Comparison: With vs Without CoT

### Without Chain of Thought

```python
from udspy import Predict

predictor = Predict(QA)
result = predictor(question="Why is the sky blue?")

print(result.answer)
# "The sky is blue due to Rayleigh scattering."
```

### With Chain of Thought

```python
cot_predictor = ChainOfThought(QA)
result = cot_predictor(question="Why is the sky blue?")

print(result.reasoning)
# "Let me explain the physics: Sunlight contains all colors. As it enters
#  the atmosphere, it interacts with air molecules. Blue light has shorter
#  wavelengths and scatters more than other colors (Rayleigh scattering).
#  This scattered blue light reaches our eyes from all directions."

print(result.answer)
# "The sky appears blue because blue light scatters more in the atmosphere
#  due to its shorter wavelength (Rayleigh scattering)."
```

**Benefits**:
- More detailed and accurate answers
- Shows the reasoning process
- Better for complex or multi-step problems
- Easier to verify correctness

## Best Practices

### 1. Use for Complex Tasks

Chain of Thought shines for tasks requiring reasoning:

```python
# Good use cases
- Math problems
- Logic puzzles
- Multi-step analysis
- Proof generation
- Planning tasks

# Less useful for
- Simple factual recall ("What is 2+2?")
- Classification without reasoning
- Direct information retrieval
```

### 2. Adjust Temperature

```python
# For deterministic tasks (math, logic)
cot = ChainOfThought(QA, temperature=0.0)

# For creative reasoning
cot = ChainOfThought(QA, temperature=0.7)
```

### 3. Review Reasoning Quality

Always check if reasoning makes sense:

```python
result = cot(question="Complex problem")

if "step" in result.reasoning.lower():
    print("✓ Good reasoning structure")

if len(result.reasoning) < 50:
    print("⚠ Reasoning might be too brief")
```

## Real-World Examples

### Code Review Reasoning

```python
class CodeReview(Signature):
    """Review code for issues."""
    code: str = InputField()
    issues: list[str] = OutputField()
    severity: str = OutputField()

reviewer = ChainOfThought(CodeReview)
result = reviewer(code="""
def divide(a, b):
    return a / b
""")

print(result.reasoning)
# "Let me analyze this code:
#  1. No error handling for division by zero
#  2. No type checking
#  3. No documentation
#  These are significant issues."

print(result.issues)
# ["Division by zero not handled", "Missing type hints", "No docstring"]

print(result.severity)
# "High - can cause runtime errors"
```

### Decision Making

```python
class Decision(Signature):
    """Make informed decisions."""
    situation: str = InputField()
    options: list[str] = InputField()
    decision: str = OutputField()
    justification: str = OutputField()

decider = ChainOfThought(Decision)
result = decider(
    situation="Need to scale database, budget is tight",
    options=["Vertical scaling", "Horizontal scaling", "Managed service"]
)

print(result.reasoning)
# "Let me evaluate each option:
#  - Vertical: Quick but limited and expensive long-term
#  - Horizontal: Complex but scalable
#  - Managed: Higher cost but less maintenance
#  Given budget constraints..."

print(result.decision)
print(result.justification)
```

See the [full example](https://github.com/silvestrid/udspy/blob/main/examples/chain_of_thought.py) in the repository.
