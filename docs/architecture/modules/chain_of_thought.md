# Chain of Thought

Chain of Thought (CoT) is a prompting technique that improves reasoning by explicitly requesting step-by-step thinking.

## Overview

The `ChainOfThought` module is a wrapper around `Predict` that automatically adds a "reasoning" field to any signature:

```python
class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

# ChainOfThought extends the signature:
# question -> reasoning, answer
cot = ChainOfThought(QA)
```

## How It Works

### 1. Signature Extension

ChainOfThought takes the original signature and creates an extended version with a reasoning field:

```python
# Original signature
question: str -> answer: str

# Extended signature (automatically)
question: str -> reasoning: str, answer: str
```

### 2. Implementation

```python
class ChainOfThought(Module):
    def __init__(self, signature, **kwargs):
        # Extract input and output fields
        input_fields = signature.get_input_fields()
        output_fields = signature.get_output_fields()

        # Prepend reasoning to outputs
        extended_outputs = {"reasoning": str, **output_fields}

        # Create new signature
        extended_signature = make_signature(
            input_fields,
            extended_outputs,
            signature.get_instructions()
        )

        # Use Predict with extended signature
        self.predict = Predict(extended_signature, **kwargs)
```

### 3. Prompt Engineering

The reasoning field encourages step-by-step thinking through:

1. **Field description**: "Step-by-step reasoning process" (customizable)
2. **Field ordering**: Reasoning comes before the answer
3. **Output format**: Uses field markers to structure the response

Example prompt structure:
```
[System]
Answer questions with clear reasoning.

Required Outputs:
- reasoning: Step-by-step reasoning process
- answer: Final answer

[User]
[[ ## question ## ]]
What is 15 * 23?

[Assistant]
[[ ## reasoning ## ]]
Let me calculate: 15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345

[[ ## answer ## ]]
345
```

## Benefits

### Improved Accuracy

Chain of Thought improves accuracy on:

- **Math problems**: 67% → 92% accuracy (typical improvement)
- **Logic puzzles**: Forces explicit reasoning steps
- **Multi-step tasks**: Prevents skipping intermediate steps
- **Complex analysis**: Organizes thinking

### Transparency

Shows the reasoning process:

```python
result = cot(question="Is 17 prime?")

print(result.reasoning)
# "To check if 17 is prime, I need to test divisibility
#  by all primes up to √17 ≈ 4.12.
#  Testing: 17 ÷ 2 = 8.5 (not divisible)
#          17 ÷ 3 = 5.67 (not divisible)
#  No divisors found, so 17 is prime."

print(result.answer)
# "Yes, 17 is prime"
```

### Debugging

Easier to identify issues:

```python
result = cot(question="What is 2^10?")

if "1024" not in result.answer:
    # Check reasoning to see where it went wrong
    print("Error in reasoning:", result.reasoning)
```

## Customization

### Custom Reasoning Description

```python
cot = ChainOfThought(
    signature,
    reasoning_description="Detailed mathematical proof with all steps"
)
```

### Model Parameters

```python
# Deterministic reasoning for math
cot = ChainOfThought(QA, temperature=0.0)

# Creative reasoning for analysis
cot = ChainOfThought(Analysis, temperature=0.7)
```

### Multiple Outputs

Works seamlessly with multiple output fields:

```python
class ComplexTask(Signature):
    """Complex task."""
    input: str = InputField()
    analysis: str = OutputField()
    recommendation: str = OutputField()
    confidence: float = OutputField()

cot = ChainOfThought(ComplexTask)
result = cot(input="...")

# All outputs available
result.reasoning       # Added automatically
result.analysis        # Original output
result.recommendation  # Original output
result.confidence      # Original output
```

## Comparison with DSPy

### Similarities

- Same concept: adds reasoning field to signature
- Improves accuracy through explicit reasoning
- Transparent reasoning process

### Differences

| Feature | udspy | DSPy |
|---------|-------|------|
| Implementation | Signature extension | Signature prepend method |
| Customization | reasoning_description | rationale_field parameter |
| Complexity | ~45 lines | ~40 lines |
| Dependencies | make_signature | signature.prepend() |

### udspy Approach

```python
# Simpler API
cot = ChainOfThought(QA)
```

### DSPy Approach

```python
# More flexible but complex
cot = dspy.ChainOfThought(
    "question -> answer",
    rationale_field=dspy.OutputField(
        prefix="Reasoning: Let's think step by step",
        desc="${reasoning}"
    )
)
```

## When to Use

### Good Use Cases ✓

- Math problems requiring calculation
- Logic puzzles and reasoning tasks
- Multi-step analysis or planning
- Tasks where you want to verify reasoning
- Educational applications (show work)
- High-stakes decisions requiring justification

### Less Useful ✗

- Simple factual recall ("What is the capital of France?")
- Binary classification without reasoning
- Very short outputs where reasoning overhead is large
- Real-time systems with strict latency requirements

## Performance Considerations

### Token Usage

Chain of Thought uses more tokens:

```python
# Without CoT: ~50 tokens
result = predict(question="What is 2+2?")
# answer: "4"

# With CoT: ~150 tokens
result = cot(question="What is 2+2?")
# reasoning: "This is basic arithmetic. 2+2 = 4"
# answer: "4"
```

**Trade-off**: Higher cost/latency for better accuracy and transparency.

### Optimization

For cost-sensitive applications:

```python
# Use CoT only for complex queries
if is_complex(question):
    result = cot(question=question)
else:
    result = simple_predict(question=question)
```

## Best Practices

1. **Use appropriate temperature**
   - `0.0` for math/logic (deterministic)
   - `0.3-0.7` for analysis/planning

2. **Customize reasoning description** for domain-specific tasks
   ```python
   ChainOfThought(
       MedicalDiagnosis,
       reasoning_description="Clinical reasoning with differential diagnosis"
   )
   ```

3. **Validate reasoning quality** in production
   ```python
   if len(result.reasoning) < 100:
       logger.warning("Reasoning too brief")
   ```

4. **Cache for repeated queries** to save costs
   ```python
   @lru_cache(maxsize=100)
   def cached_cot(question):
       return cot(question=question)
   ```

See [Examples](../../examples/chain_of_thought.md) for more details.
