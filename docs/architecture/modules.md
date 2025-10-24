# Modules

Modules are composable units that encapsulate LLM calls.

## Predict Module

The core module is `Predict`, which maps inputs to outputs via an LLM:

```python
from udspy import Predict

predictor = Predict(
    signature=QA,
    model="gpt-4o-mini",
    temperature=0.7,
)

result = predictor(question="What is AI?")
```

## Custom Modules

Create custom modules by subclassing `Module`:

```python
from udspy import Module, Predict, Prediction

class ChainOfThought(Module):
    def __init__(self, signature):
        self.think = Predict(make_signature(
            signature.get_input_fields(),
            {"reasoning": str},
            "Think step by step",
        ))
        self.answer = Predict(signature)

    def forward(self, **inputs):
        # First, generate reasoning
        thought = self.think(**inputs)

        # Then, generate answer with reasoning
        result = self.answer(**inputs, reasoning=thought.reasoning)

        return result
```

## Composition

Modules can be composed to build complex behaviors:

```python
class Pipeline(Module):
    def __init__(self):
        self.analyze = Predict(AnalysisSignature)
        self.summarize = Predict(SummarySignature)

    def forward(self, text):
        analysis = self.analyze(text=text)
        summary = self.summarize(
            text=text,
            analysis=analysis.result,
        )
        return Prediction(
            analysis=analysis.result,
            summary=summary.result,
        )
```

See [API: Modules](../api/module.md) for detailed documentation.
