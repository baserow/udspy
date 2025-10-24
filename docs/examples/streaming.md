# Streaming Examples

Learn how to use streaming for better user experience.

## Basic Streaming

```python
import asyncio
from udspy import StreamingPredict, Signature, InputField, OutputField

class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

async def main():
    predictor = StreamingPredict(QA)

    async for chunk in predictor.stream(question="What is AI?"):
        if isinstance(chunk, StreamChunk):
            print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

## Multi-Field Streaming

Stream reasoning and answer separately:

```python
class ReasonedQA(Signature):
    """Answer with reasoning."""
    question: str = InputField()
    reasoning: str = OutputField()
    answer: str = OutputField()

async def main():
    predictor = StreamingPredict(ReasonedQA)

    print("Question: What is the sum of first 10 primes?\n")

    async for item in predictor.stream(
        question="What is the sum of first 10 primes?"
    ):
        if isinstance(item, StreamChunk):
            if item.field_name == "reasoning":
                print(f"💭 {item.delta}", end="", flush=True)
            elif item.field_name == "answer":
                print(f"\n✓ {item.delta}", end="", flush=True)

            if item.is_complete:
                print()  # Newline after field completes

        elif isinstance(item, Prediction):
            print(f"\n\nFinal: {item.answer}")

asyncio.run(main())
```

## Web Application Integration

Use streaming in a web application:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/ask")
async def ask_question(question: str):
    async def generate():
        predictor = StreamingPredict(QA)
        async for chunk in predictor.stream(question=question):
            if isinstance(chunk, StreamChunk) and not chunk.is_complete:
                # chunk.delta contains the new incremental text
                # chunk.content contains the full accumulated text so far
                yield f"data: {chunk.delta}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

See the [full example](../../examples/streaming_example.py) in the repository.
