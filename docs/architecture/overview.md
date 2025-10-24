# Architecture Overview

udspy consists of four main components that work together to provide a clean abstraction for LLM interactions.

## Core Components

### 1. Signatures

Signatures define the input/output contract for an LLM task using Pydantic models.

```python
class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()
```

**Key responsibilities:**
- Define input and output fields with type information
- Provide field descriptions for prompt construction
- Validate data at runtime using Pydantic

[Learn more about Signatures →](signatures.md)

### 2. Adapters

Adapters handle the formatting of signatures into LLM-specific message formats and parsing responses back.

```python
adapter = ChatAdapter()
instructions = adapter.format_instructions(signature)
formatted_input = adapter.format_inputs(signature, inputs)
outputs = adapter.parse_outputs(signature, completion)
```

**Key responsibilities:**
- Convert signature definitions to system prompts
- Format input values into user messages
- Parse LLM completions into structured outputs
- Convert Pydantic models to OpenAI tool schemas

[Learn more about Adapters →](adapters.md)

### 3. Modules

Modules are composable units that encapsulate LLM calls. The core module is `Predict`.

```python
predictor = Predict(signature, model="gpt-4o-mini")
result = predictor(question="What is AI?")
```

**Key responsibilities:**
- Manage signature, model, and configuration
- Orchestrate adapter formatting and API calls
- Return structured `Prediction` objects
- Support composition and reuse

[Learn more about Modules →](modules.md)

### 4. Streaming

Streaming support allows incremental output processing for better UX.

```python
predictor = StreamingPredict(signature)
async for chunk in predictor.stream(question="Explain AI"):
    print(chunk.content, end="", flush=True)
```

**Key responsibilities:**
- Process streaming API responses incrementally
- Detect field boundaries in streams
- Emit field-specific chunks
- Provide final parsed prediction

[Learn more about Streaming →](streaming.md)

## Data Flow

```
User Input → Signature → Adapter → OpenAI API → Adapter → Prediction → User
              ↓           ↓                        ↑
           Validate    Format                   Parse
```

1. User provides input matching signature's input fields
2. Signature validates input types
3. Adapter formats signature + inputs into messages
4. OpenAI API generates completion
5. Adapter parses completion into structured outputs
6. User receives `Prediction` with typed outputs

## Design Principles

### Native Tool Calling

Unlike DSPy which uses custom field markers in prompts, udspy uses OpenAI's native function calling:

```python
class Calculator(BaseModel):
    """Perform arithmetic."""
    operation: str
    a: float
    b: float

predictor = Predict(signature, tools=[Calculator])
```

This provides:
- Better performance (optimized by OpenAI)
- More reliable parsing
- Cleaner prompts
- Forward compatibility

### Minimal Abstractions

Every component has a clear, focused responsibility:

- **Signatures**: Define I/O contracts
- **Adapters**: Handle format translation
- **Modules**: Encapsulate LLM calls
- **Streaming**: Process incremental outputs

No hidden magic, no over-engineering.

### Type Safety

Pydantic provides runtime type checking throughout:

```python
class QA(Signature):
    question: str = InputField()
    answer: int = OutputField()  # Will validate output is int

predictor = Predict(QA)
result = predictor(question="What is 2+2?")
assert isinstance(result.answer, int)
```

## Configuration

Global configuration via `settings`:

```python
import udspy

udspy.settings.configure(
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.7,
)
```

Per-module overrides:

```python
predictor = Predict(
    signature,
    model="gpt-4",
    temperature=0.0,
)
```

## Next Steps

- Deep dive into [Signatures](signatures.md)
- Understand [Adapters](adapters.md)
- Explore [Modules](modules.md)
- Learn about [Streaming](streaming.md)
