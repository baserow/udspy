# Streaming

Streaming support allows incremental processing of LLM outputs.

## Overview

Streaming provides better user experience by showing results as they're generated:

```python
from udspy import StreamingPredict

predictor = StreamingPredict(signature)

async for chunk in predictor.stream(question="Explain AI"):
    if isinstance(chunk, StreamChunk):
        print(chunk.content, end="", flush=True)
```

## StreamChunk

Each chunk contains:

- `field_name`: Which output field this chunk belongs to
- `content`: Incremental content
- `is_complete`: Whether the field is finished

```python
async for item in predictor.stream(**inputs):
    if isinstance(item, StreamChunk):
        print(f"[{item.field_name}] {item.content}", end="")
        if item.is_complete:
            print(f"\n--- {item.field_name} complete ---")
    elif isinstance(item, Prediction):
        print(f"\nFinal result: {item}")
```

## Converting Predictors

Convert any `Predict` to streaming:

```python
from udspy import Predict, streamify

predictor = Predict(signature, temperature=0.7)
streaming_predictor = streamify(predictor)

async for chunk in streaming_predictor.stream(**inputs):
    ...
```

## Field-Specific Streaming

Streaming automatically detects field boundaries:

```python
class ReasonedQA(Signature):
    """Answer with reasoning."""
    question: str = InputField()
    reasoning: str = OutputField()
    answer: str = OutputField()

# Will emit chunks for 'reasoning' then 'answer'
async for chunk in predictor.stream(question="Why is sky blue?"):
    if chunk.field_name == "reasoning":
        print(f"Thinking: {chunk.content}", end="")
    elif chunk.field_name == "answer":
        print(f"Answer: {chunk.content}", end="")
```

See [API: Streaming](../api/streaming.md) and [Examples: Streaming](../examples/streaming.md) for more details.
