# udspy Architecture Overview

This document defines the architecture, design principles, and responsibility boundaries for udspy. Use this as the **authoritative reference** when deciding where to place new code or how to extend the library.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Layered Architecture](#layered-architecture)
3. [Core Abstractions](#core-abstractions)
4. [Module System](#module-system)
5. [LM Abstraction (Language Model Layer)](#lm-abstraction-language-model-layer)
6. [Signatures](#signatures)
7. [Tools](#tools)
8. [History](#history)
9. [Streaming](#streaming)
10. [Confirmation & Suspend/Resume](#confirmation--suspendresume)
11. [Callbacks](#callbacks)
12. [How to Extend](#how-to-extend)
13. [Design Patterns](#design-patterns)
14. [Decision Tree](#decision-tree)

---

## Core Philosophy

udspy is a **minimal, async-first framework** for building LLM applications with clear abstractions and separation of concerns.

### Key Principles

1. **Simplicity Over Completeness**
   - Provide core primitives, not every possible feature
   - Make common cases easy, complex cases possible

2. **Async-First**
   - All core operations are async
   - Sync wrappers (`forward()`, `__call__()`) use `asyncio.run()` internally
   - Natural support for streaming and concurrent operations

3. **Clear Responsibility Boundaries**
   - Each layer has ONE well-defined purpose
   - Minimal coupling between layers
   - Easy to test and modify independently

4. **Type Safety**
   - Pydantic models for runtime validation
   - Type hints throughout
   - Fail fast with clear errors

5. **Native Tool Calling**
   - Use OpenAI's native function calling API
   - No custom prompt hacking for structured outputs
   - Leverages provider optimizations

---

## Layered Architecture

udspy is organized into clear layers with well-defined responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Module Layer                           │
│  (Predict, ChainOfThought, ReAct, Custom Modules)       │
│  - Business logic and orchestration                      │
│  - Compose other modules                                 │
│  - Handle tool execution loops                           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  LM Layer (Provider Abstraction)         │
│  - Abstract interface to LLM providers                   │
│  - Currently: OpenAI via settings.lm                     │
│  - Extensible: Anthropic, local models, etc.             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Adapter Layer                          │
│  - Format signatures → messages                          │
│  - Parse LLM outputs → structured data                   │
│  - Convert tools → provider schemas                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Supporting Infrastructure                   │
│  - History: Conversation state                           │
│  - Tools: Function calling                               │
│  - Streaming: Event queue and chunks                     │
│  - Confirmation: Human-in-the-loop                       │
│  - Callbacks: Telemetry and monitoring                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    Settings                              │
│  - Global and context-specific configuration             │
│  - Thread-safe via ContextVar                            │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### 1. Module Layer (`src/udspy/module/`)
**What it does**: Orchestrates LLM calls and business logic

**Responsibilities**:
- Implements business logic (Predict, ChainOfThought, ReAct)
- Composes other modules
- Manages execution flow (tool loops, retry logic)
- Emits streaming events
- Returns final Prediction results

**Key Files**:
- `base.py` - Module base class with aexecute/aforward/astream
- `predict.py` - Core LLM prediction with tool calling
- `chain_of_thought.py` - Reasoning wrapper
- `react.py` - Agent with tool iteration

**DO**: Business logic, orchestration, composition
**DON'T**: Direct LLM API calls (use LM layer), message formatting (use Adapter)

---

#### 2. LM Layer (Language Model Abstraction)
**What it does**: Provides abstract interface to LLM providers

**Current State**:
- Direct usage of `settings.aclient` (AsyncOpenAI)
- No abstraction yet - coupled to OpenAI

**Future Design**:
```python
class LM(ABC):
    """Abstract language model interface."""

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[ChatCompletion, None] | ChatCompletion:
        """Complete a prompt with optional tools."""
        pass

class OpenAILM(LM):
    """OpenAI implementation."""
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def acomplete(self, messages, tools=None, stream=False, **kwargs):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=stream,
            **kwargs
        )
```

**Responsibilities**:
- Normalize provider-specific APIs
- Handle retries and rate limiting
- Abstract away provider differences
- Provide unified interface for modules

**Key Files** (when implemented):
- `lm.py` - Base LM class and interface
- `lm/openai.py` - OpenAI implementation
- `lm/anthropic.py` - Anthropic implementation (future)

**DO**: Provider API calls, retry logic, rate limiting
**DON'T**: Message formatting (use Adapter), business logic (use Module)

---

#### 3. Adapter Layer (`src/udspy/adapter.py`)
**What it does**: Translates between udspy concepts and provider formats

**Responsibilities**:
- Format Signature → system prompts
- Format inputs → user messages
- Parse LLM outputs → structured Prediction
- Convert Tool → OpenAI tool schema
- Type coercion and validation

**Key Methods**:
- `format_instructions(signature)` - Signature → system message
- `format_inputs(signature, inputs)` - Inputs → user message
- `parse_outputs(signature, completion)` - Completion → structured dict
- `tools_to_openai_format(tools)` - Tools → OpenAI schemas

**DO**: Format translation, schema conversion, parsing
**DON'T**: LLM API calls (use LM), orchestration (use Module)

---

#### 4. Supporting Infrastructure

**History** (`src/udspy/history.py`)
- Stores conversation messages
- Simple list wrapper with convenience methods
- No LLM coupling - pure data structure

**Tools** (`src/udspy/tool.py`)
- Wraps functions as tools
- Extracts schemas from type hints
- Handles async/sync execution
- Integrates with confirmation system

**Streaming** (`src/udspy/streaming.py`)
- Event queue via ContextVar
- StreamEvent base class
- emit_event() for custom events
- Prediction as final event

**Confirmation** (`src/udspy/confirmation.py`)
- ConfirmationRequired exception
- ResumeState for continuation
- Context-based approval tracking
- @confirm_first decorator

**Callbacks** (`src/udspy/callback.py`)
- BaseCallback interface
- @with_callbacks decorator
- Telemetry and monitoring hooks
- Compatible with Opik, MLflow, etc.

---

## Core Abstractions

### 1. Module

**What**: A composable unit that encapsulates LLM operations

**Interface**:
```python
class Module:
    async def aexecute(self, *, stream: bool = False, **inputs) -> Prediction:
        """Core execution - implements business logic."""

    async def aforward(self, **inputs) -> Prediction:
        """Non-streaming execution."""
        return await self.aexecute(stream=False, **inputs)

    async def astream(self, **inputs) -> AsyncGenerator[StreamEvent, None]:
        """Streaming execution - sets up queue and yields events."""

    def forward(self, **inputs) -> Prediction:
        """Sync wrapper."""
        return asyncio.run(self.aforward(**inputs))

    def __call__(self, **inputs) -> Prediction:
        """Sync convenience."""
        return self.forward(**inputs)
```

**Key Insight**: `aexecute()` is the **single source of truth**. Both `aforward()` and `astream()` call it with different parameters.

---

### 2. Signature

**What**: Defines input/output contract for an LLM task using Pydantic

**Purpose**:
- Specify expected inputs and outputs
- Provide descriptions for prompt construction
- Enable runtime validation
- Generate tool schemas

**Example**:
```python
class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Concise answer")
```

**Where Used**:
- Modules use signatures to define their I/O contract
- Adapter formats signatures into prompts
- Validation ensures type safety

---

### 3. Prediction

**What**: Result of a module execution (dict-like with attribute access)

```python
pred = Prediction(answer="Paris", reasoning="France's capital")
print(pred.answer)  # "Paris"
print(pred["answer"])  # "Paris"
print(pred.is_final())  # True if no pending tool calls
```

**Key Properties**:
- `native_tool_calls` - Pending tool calls (if any)
- `is_final()` - True if execution is complete
- Inherits from `dict` and `StreamEvent`

---

### 4. Tool

**What**: Wrapper for a callable function that can be invoked by LLM

**Creation**:
```python
@tool(name="search", description="Search the web")
def search(query: str = Field(description="Search query")) -> str:
    return search_web(query)

# Or manually
search_tool = Tool(
    func=search_fn,
    name="search",
    description="Search the web",
    require_confirmation=True
)
```

**Schema Generation**:
- Extracts type hints from function signature
- Uses Pydantic Field for parameter descriptions
- Converts to OpenAI function calling format

---

### 5. History

**What**: Conversation message storage

**Usage**:
```python
history = History()

# Automatically managed by Predict
result = predictor(question="What is Python?", history=history)
# History now contains: [system, user, assistant]

result = predictor(question="What are its features?", history=history)
# LLM has context from previous turn
```

**Storage Format**: OpenAI message format (`{"role": "...", "content": "..."}`)

---

## Module System

### The `aexecute` Pattern

**Core Concept**: Every module has ONE implementation in `aexecute()`, which powers BOTH streaming and non-streaming interfaces.

```python
class MyModule(Module):
    async def aexecute(self, *, stream: bool = False, **inputs) -> Prediction:
        """Single source of truth for execution logic."""

        # 1. Do the work (call LLM, process data, etc.)
        result = await self.do_work(inputs, stream=stream)

        # 2. Optionally emit streaming events
        if self.should_emit_events():
            await emit_event(OutputStreamChunk(...))

        # 3. Always return final Prediction
        return Prediction(answer=result)
```

### How Streaming Works

**Event Queue**:
- `astream()` sets up an `asyncio.Queue` via ContextVar
- Modules emit events using `await emit_event(event)`
- Queue is automatically available to nested modules

**Flow**:
```
User calls module.astream()
    ↓
astream() creates queue, sets in ContextVar
    ↓
astream() calls aexecute(stream=True)
    ↓
aexecute() does work, emits events via emit_event()
    ↓
astream() yields events from queue
    ↓
Final Prediction is yielded
```

**Example**:
```python
async for event in predictor.astream(question="What is AI?"):
    if isinstance(event, OutputStreamChunk):
        print(event.delta, end="", flush=True)  # Real-time output
    elif isinstance(event, Prediction):
        result = event  # Final result
```

### How Non-Streaming Works

**Simple**:
- `aforward()` calls `aexecute(stream=False)`
- No queue is set up
- Events are not emitted (or silently ignored)
- Only final Prediction is returned

```python
result = await predictor.aforward(question="What is AI?")
print(result.answer)  # Just the final answer
```

### Composing Modules

Modules can contain other modules:

```python
class Pipeline(Module):
    def __init__(self):
        self.step1 = Predict(Signature1)
        self.step2 = ChainOfThought(Signature2)

    async def aexecute(self, *, stream: bool = False, **inputs):
        # Get result from first module (don't stream intermediate steps)
        result1 = await self.step1.aforward(**inputs)

        # Stream final module if requested
        result2 = await self.step2.aforward(
            input=result1.output,
            stream=stream  # Pass down stream parameter
        )

        return Prediction(final=result2.answer)
```

**Key Pattern**: Nested modules automatically emit to the active queue if one exists.

---

## LM Abstraction (Language Model Layer)

### Overview

The LM (Language Model) abstraction provides a **provider-agnostic interface** for interacting with LLMs. This allows udspy to work with different providers (OpenAI, Anthropic, local models, etc.) through a common interface.

**Location**: `src/udspy/lm/`

**Key Files**:
- `lm/base.py` - Abstract LM interface
- `lm/openai.py` - OpenAI implementation
- `lm/__init__.py` - Public API exports

### Interface

```python
from abc import ABC, abstractmethod

class LM(ABC):
    """Abstract language model interface."""

    @abstractmethod
    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        """Generate completion from the language model.

        Args:
            messages: List of messages in OpenAI format
            model: Model identifier (e.g., "gpt-4o")
            tools: Optional list of tool schemas in OpenAI format
            stream: If True, return an async generator of chunks
            **kwargs: Provider-specific parameters

        Returns:
            If stream=False: Completion response object
            If stream=True: AsyncGenerator yielding chunks
        """
        pass
```

**Design Decisions**:
- Single method interface - simple and focused
- OpenAI message format as standard (widely adopted)
- Generic return types to support any provider
- Provider implementations handle format conversion internally

### OpenAI Implementation

```python
from openai import AsyncOpenAI
from udspy.lm import OpenAILM

# Create instance
client = AsyncOpenAI(api_key="sk-...")
lm = OpenAILM(client, default_model="gpt-4o")

# Use directly
response = await lm.acomplete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

**Features**:
- Wraps AsyncOpenAI client
- Supports default model (optional, can override per call)
- Passes through all OpenAI parameters
- Handles both streaming and non-streaming

### Settings Integration

The LM abstraction integrates with udspy's settings system:

```python
import udspy

# Configure with API key (creates OpenAILM automatically)
udspy.settings.configure(api_key="sk-...", model="gpt-4o")

# Access the LM
lm = udspy.settings.lm  # Returns OpenAILM instance

# Or provide custom LM
from udspy.lm import OpenAILM
custom_lm = OpenAILM(client, default_model="gpt-4o")
udspy.settings.configure(lm=custom_lm)
```

**Backward Compatibility**: `settings.aclient` still works but is deprecated. Use `settings.lm` for new code.

### Context Manager Support

LM instances can be overridden per-context:

```python
# Global settings
udspy.settings.configure(api_key="global-key", model="gpt-4o-mini")

# Temporary override
with udspy.settings.context(lm=custom_lm):
    result = predictor(question="...")  # Uses custom_lm

# Back to global LM
result = predictor(question="...")  # Uses global LM
```

**Priority**:
1. Explicit `lm` parameter (highest)
2. `aclient` parameter (creates OpenAILM wrapper)
3. `api_key` parameter (creates new client + LM)
4. Global settings (fallback)

### Usage in Predict Module

The Predict module accesses LMs via `settings.lm`:

```python
# Non-streaming
response = await settings.lm.acomplete(
    messages=messages,
    model=model or settings.default_model,
    tools=tool_schemas,
    stream=False,
    **kwargs
)

# Streaming
stream = await settings.lm.acomplete(
    messages=messages,
    model=model or settings.default_model,
    tools=tool_schemas,
    stream=True,
    **kwargs
)
```

This centralizes all LLM calls and makes provider swapping trivial.

### Implementing Custom Providers

To add a new provider, implement the `LM` interface:

```python
from udspy.lm import LM

class AnthropicLM(LM):
    """Anthropic Claude implementation."""

    def __init__(self, api_key: str, default_model: str | None = None):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any | AsyncGenerator[Any, None]:
        actual_model = model or self.default_model
        if not actual_model:
            raise ValueError("No model specified")

        # Convert OpenAI format to Anthropic format
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Call Anthropic API
        return await self.client.messages.create(
            model=actual_model,
            messages=anthropic_messages,
            tools=anthropic_tools,
            stream=stream,
            **kwargs
        )

    def _convert_messages(self, messages):
        """Convert OpenAI → Anthropic format."""
        # Implementation details...
```

**Usage**:
```python
from my_providers import AnthropicLM

lm = AnthropicLM(api_key="sk-ant-...", default_model="claude-3-5-sonnet")
udspy.settings.configure(lm=lm)

# All udspy features work with your custom provider!
```

### Message Format Standard

LM implementations should accept/return **OpenAI message format**:

```python
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"},
]
```

**Why OpenAI format?**
- Industry standard
- Simple and flexible
- Easy to convert to other formats
- Well-documented

Custom providers convert internally.

### Responsibility Boundary

**LM Layer Owns**:
- Making API calls to providers
- Handling streaming vs non-streaming responses
- Provider-specific parameter passing
- Format conversion (provider ↔ OpenAI format)

**LM Layer Does NOT Own**:
- Prompt formatting (Adapter Layer)
- Output parsing (Adapter Layer)
- Tool execution (Module Layer)
- Retry/error handling (Module Layer)
- Orchestration logic (Module Layer)

### Related Documentation

See [LM Abstraction](lm.md) for comprehensive documentation including:
- Detailed API reference
- Custom provider implementation guide
- Context manager examples
- Type handling
- Best practices

---

## Signatures

### Purpose

Signatures define the **contract** between user inputs and LLM outputs.

### Anatomy

```python
class QA(Signature):
    """Answer questions concisely."""  # ← Instructions (docstring)

    question: str = InputField(description="User's question")  # ← Input
    answer: str = OutputField(description="Concise answer")    # ← Output
```

### How They Work

1. **Definition**: User defines Signature with InputField/OutputField
2. **Validation**: SignatureMeta validates all fields are marked
3. **Formatting**: Adapter converts to system prompt:
   ```
   Answer questions concisely.

   Inputs:
   - question (str): User's question

   Outputs:
   - answer (str): Concise answer
   ```
4. **Parsing**: Adapter parses LLM output into structured dict
5. **Return**: Module returns Prediction with typed attributes

### Field Extraction

```python
# Get inputs
QA.get_input_fields()  # {"question": FieldInfo(...)}

# Get outputs
QA.get_output_fields()  # {"answer": FieldInfo(...)}

# Get instructions
QA.get_instructions()  # "Answer questions concisely."
```

### Dynamic Creation

```python
# String format (all fields are str)
QA = Signature.from_string("question -> answer", "Answer questions")

# Programmatic (custom types)
QA = make_signature(
    input_fields={"question": str},
    output_fields={"answer": str},
    instructions="Answer questions"
)
```

### Where Signatures Live

**Module Layer**: Modules accept signatures to define I/O
**Adapter Layer**: Formats signatures into prompts
**NOT in LM Layer**: LM layer only sees formatted messages

---

## Tools

### Lifecycle

```
1. Definition (by user)
   ↓
2. Schema Extraction (Tool.__init__)
   ↓
3. Schema Conversion (Adapter.tools_to_openai_format)
   ↓
4. LLM Call (Module → LM)
   ↓
5. LLM Returns Tool Calls
   ↓
6. Tool Execution (Module calls Tool.acall)
   ↓
7. Result Formatting (back to messages)
   ↓
8. Loop until final answer
```

### Definition

```python
from pydantic import Field
from udspy import tool

@tool(name="calculator", description="Perform arithmetic")
def calculator(
    operation: str = Field(description="add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Perform arithmetic operations."""
    ops = {"add": a + b, "subtract": a - b, ...}
    return ops[operation]
```

### Schema Extraction

Tool class extracts schema from function signature:

```python
# Automatic schema generation
tool.get_args_schema() →
{
    "type": "object",
    "properties": {
        "operation": {"type": "string", "description": "add, subtract, ..."},
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    },
    "required": ["operation", "a", "b"]
}
```

### Conversion to OpenAI Format

Adapter converts Tool → OpenAI schema:

```python
{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform arithmetic",
        "parameters": {tool.get_args_schema()}
    }
}
```

### Execution

```python
# In Predict module:
async def _execute_tool_calls(self, tool_calls, tools):
    results = []
    for tc in tool_calls:
        tool = self._find_tool(tc.function.name, tools)
        result = await tool.acall(**tc.arguments)
        results.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return results
```

### With Confirmation

```python
@tool(require_confirmation=True)
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

# Raises ConfirmationRequired when called
# Module catches, saves state, waits for user approval
# Resumes execution after approval
```

---

## History

### Purpose

Store conversation context for multi-turn interactions.

### Structure

Simple wrapper around `list[dict[str, Any]]` with convenience methods:

```python
class History:
    messages: list[dict[str, Any]]

    def add_user_message(self, content: str)
    def add_assistant_message(self, content: str, tool_calls: list | None)
    def add_tool_result(self, tool_call_id: str, content: str)
    def add_system_message(self, content: str)
```

### Usage

```python
history = History()

# First turn
result = predictor(question="What is Python?", history=history)
# history.messages = [
#     {"role": "system", "content": "...instructions..."},
#     {"role": "user", "content": "question: What is Python?"},
#     {"role": "assistant", "content": "answer: A programming language"}
# ]

# Second turn (context preserved)
result = predictor(question="What are its features?", history=history)
# LLM sees full conversation history
```

### How Predict Uses History

```python
def _build_initial_messages(self, signature, inputs, history):
    messages = []

    # Add history messages if provided
    if history:
        messages.extend(history.messages)
    else:
        # Add system message with instructions
        messages.append({
            "role": "system",
            "content": self.adapter.format_instructions(signature)
        })

    # Add current input
    messages.append({
        "role": "user",
        "content": self.adapter.format_inputs(signature, inputs)
    })

    return messages
```

### When to Use

- **Multi-turn conversations**: Chatbots, assistants
- **Context-dependent tasks**: "It" and "that" references
- **Iterative refinement**: Follow-up questions

### When NOT to Use

- **Stateless tasks**: One-off questions
- **Independent requests**: No cross-request context needed

---

## Streaming

### Architecture

**Event Queue**: ContextVar-based queue for thread-safe event emission

```python
_stream_queue: ContextVar[asyncio.Queue | None] = ContextVar("_stream_queue", default=None)

async def emit_event(event: StreamEvent):
    """Emit event to active stream (if any)."""
    queue = _stream_queue.get()
    if queue is not None:
        await queue.put(event)
```

### Event Types

**Base Class**:
```python
class StreamEvent:
    """Base for all events."""
    pass
```

**Built-in Events**:
- `OutputStreamChunk` - LLM output for a field
- `ThoughtStreamChunk` - Reasoning/thought output
- `Prediction` - Final result

**Custom Events**:
```python
from dataclasses import dataclass

@dataclass
class ToolProgress(StreamEvent):
    tool_name: str
    progress: float
    message: str

# Emit from anywhere
await emit_event(ToolProgress("search", 0.5, "Searching..."))
```

### Flow

```
1. User calls module.astream()
   ↓
2. astream() creates Queue, sets in _stream_queue ContextVar
   ↓
3. astream() spawns task: aexecute(stream=True)
   ↓
4. aexecute() does work, calls emit_event() for chunks
   ↓
5. emit_event() puts events in queue
   ↓
6. astream() yields events from queue
   ↓
7. Final Prediction is yielded
```

### Chunk Structure

```python
@dataclass
class OutputStreamChunk(StreamEvent):
    module: Module          # Which module emitted this
    field_name: str         # Which output field
    delta: str              # New content since last chunk
    content: str            # Full accumulated content so far
    is_complete: bool       # Is this field done?
```

### Usage Pattern

```python
async for event in predictor.astream(question="Explain AI"):
    if isinstance(event, OutputStreamChunk):
        if event.field_name == "answer":
            print(event.delta, end="", flush=True)
    elif isinstance(event, Prediction):
        result = event
        print(f"\n\nFinal: {result.answer}")
```

### Nested Modules

Events from nested modules bubble up automatically:

```python
class Pipeline(Module):
    async def aexecute(self, *, stream: bool = False, **inputs):
        # Nested module emits to same queue
        async for event in self.predictor.aexecute(stream=stream, **inputs):
            # Events automatically go to active queue
            if isinstance(event, Prediction):
                return event
```

---

## Confirmation & Suspend/Resume

### Purpose

Enable **human-in-the-loop** patterns where execution pauses for user input.

### Flow

```
1. Tool/Module raises ConfirmationRequired
   ↓
2. Exception propagates to user code
   ↓
3. User sees question, responds
   ↓
4. User creates ResumeState(exception, response)
   ↓
5. User calls module.aforward(resume_state=resume_state)
   ↓
6. Module resumes execution with user response
   ↓
7. Execution completes
```

### Exceptions

```python
class ConfirmationRequired(Exception):
    question: str                    # What to ask user
    confirmation_id: str             # Unique ID
    tool_call: ToolCall | None       # If raised by tool
    context: dict[str, Any]          # Module-specific state

class ConfirmationRejected(Exception):
    message: str                     # Why rejected
    confirmation_id: str             # Which confirmation
    tool_call: ToolCall | None       # If raised by tool
```

### Resume State

```python
class ResumeState:
    exception: ConfirmationRequired  # Original exception
    user_response: str               # User's answer ("yes", "no", JSON, etc.)
```

### Usage: Loop Pattern

```python
from udspy import ResumeState

resume_state = None

while True:
    try:
        result = agent(
            question="Delete all files",
            resume_state=resume_state
        )
        break  # Success
    except ConfirmationRequired as e:
        print(f"\n{e.question}")
        user_input = input("Approve? (yes/no): ")
        resume_state = ResumeState(e, user_input)
```

### Usage: Streaming Pattern

```python
async for event in agent.astream(question="Delete files"):
    if isinstance(event, Prediction):
        if not event.is_final():
            # Has pending tool calls requiring confirmation
            for tc in event.native_tool_calls:
                # Show confirmation UI
                approved = await ask_user(tc)

                # Resume with response
                resume_state = ResumeState(exception, "yes" if approved else "no")

                # Continue streaming
                async for event2 in agent.astream(resume_state=resume_state):
                    yield event2
```

### Module Implementation

Modules that support suspend/resume must implement:

```python
class MyModule(Module):
    async def aexecute(self, *, stream: bool = False, resume_state=None, **inputs):
        # Check for resume
        if resume_state:
            return await self.aresume(
                user_response=resume_state.user_response,
                saved_state=resume_state.exception.context
            )

        # Normal execution
        try:
            result = await self.do_work()
            return Prediction(**result)
        except ConfirmationRequired as e:
            # Save state in exception context
            e.context["saved_data"] = self.state
            raise  # Let user handle

    async def aresume(self, user_response: str, saved_state: dict):
        # Restore state
        self.state = saved_state["saved_data"]

        # Process user response
        if user_response == "yes":
            # Continue
            return await self.do_work()
        else:
            # Abort
            raise ConfirmationRejected("User rejected")
```

### Decorator for Tools

```python
from udspy import tool, confirm_first

@tool(require_confirmation=True)
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

# Or manually
@confirm_first
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"
```

---

## Callbacks

### Purpose

Provide hooks for telemetry, monitoring, and observability.

### Interface

```python
class BaseCallback:
    def on_module_start(self, call_id: str, instance: Module, inputs: dict):
        """Called when module execution starts."""
        pass

    def on_module_end(self, call_id: str, outputs: Any, exception: Exception | None):
        """Called when module execution ends."""
        pass

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict):
        """Called when LLM call starts."""
        pass

    def on_lm_end(self, call_id: str, outputs: Any, exception: Exception | None):
        """Called when LLM call ends."""
        pass

    def on_tool_start(self, call_id: str, instance: Tool, inputs: dict):
        """Called when tool execution starts."""
        pass

    def on_tool_end(self, call_id: str, outputs: Any, exception: Exception | None):
        """Called when tool execution ends."""
        pass
```

### Usage

```python
class LoggingCallback(BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        print(f"[{call_id}] LLM called")
        print(f"  Model: {inputs.get('model')}")
        print(f"  Messages: {len(inputs.get('messages', []))}")

    def on_lm_end(self, call_id, outputs, exception):
        if exception:
            print(f"[{call_id}] LLM failed: {exception}")
        else:
            print(f"[{call_id}] LLM completed")

# Configure globally
udspy.settings.configure(callbacks=[LoggingCallback()])

# Or per-context
with udspy.settings.context(callbacks=[LoggingCallback()]):
    result = predictor(question="...")
```

### Integration

Callbacks are triggered by `@with_callbacks` decorator:

```python
@with_callbacks
async def aexecute(self, *, stream: bool = False, **inputs):
    # Callbacks automatically called before/after
    pass
```

### Compatibility

Compatible with:
- **Opik**: MLOps platform for LLM applications
- **MLflow**: ML experiment tracking
- **Custom**: Any monitoring system

---

## How to Extend

### Adding a New Module

1. **Subclass Module**
2. **Implement aexecute()**
3. **Emit events** (if streaming)
4. **Return Prediction**

```python
from udspy import Module, Prediction, OutputStreamChunk, emit_event

class MyModule(Module):
    def __init__(self, signature):
        self.signature = signature
        self.predictor = Predict(signature)

    async def aexecute(self, *, stream: bool = False, **inputs):
        # Custom pre-processing
        processed = self.preprocess(inputs)

        # Call nested module
        result = await self.predictor.aforward(**processed)

        # Custom post-processing
        final = self.postprocess(result)

        # Emit event if streaming
        if stream:
            await emit_event(OutputStreamChunk(
                module=self,
                field_name="answer",
                delta=final["answer"],
                content=final["answer"],
                is_complete=True
            ))

        # Return final prediction
        return Prediction(**final)
```

### Adding a New LM Provider

1. **Create src/udspy/lm/ package**
2. **Define LM base class**
3. **Implement provider-specific class**
4. **Update settings to use LM**

```python
# src/udspy/lm/base.py
from abc import ABC, abstractmethod

class LM(ABC):
    @abstractmethod
    async def acomplete(self, messages, *, tools=None, stream=False, **kwargs):
        pass

# src/udspy/lm/anthropic.py
class AnthropicLM(LM):
    def __init__(self, client, model):
        self.client = client
        self.model = model

    async def acomplete(self, messages, *, tools=None, stream=False, **kwargs):
        # Convert formats
        # Call Anthropic API
        # Convert response
        pass
```

### Adding Custom Stream Events

```python
from dataclasses import dataclass
from udspy.streaming import StreamEvent, emit_event

@dataclass
class CustomEvent(StreamEvent):
    message: str
    progress: float

# Emit from anywhere
async def my_function():
    await emit_event(CustomEvent("Processing...", 0.5))
```

### Adding Custom Tool Logic

```python
from udspy import Tool
from pydantic import Field

@tool(name="custom_tool")
async def custom_tool(param: str = Field(...)) -> str:
    """Custom tool with async logic."""
    result = await async_operation(param)
    return result
```

---

## Design Patterns

### 1. Async-First, Sync Wrappers

**Pattern**: All core logic is async, sync is a wrapper

```python
async def aforward(self, **inputs) -> Prediction:
    """Async implementation."""
    return await self.do_async_work(inputs)

def forward(self, **inputs) -> Prediction:
    """Sync wrapper."""
    ensure_sync_context("forward")
    return asyncio.run(self.aforward(**inputs))
```

**Why**: Async is more flexible, can't go async→sync→async

---

### 2. Single Execution Path

**Pattern**: One `aexecute()` implementation for both streaming and non-streaming

```python
async def aexecute(self, *, stream: bool = False, **inputs):
    # Check if should stream
    if stream and _stream_queue.get() is not None:
        await emit_event(chunk)

    # Always return final result
    return Prediction(...)
```

**Why**: DRY, easier to maintain, composable

---

### 3. Event Queue via ContextVar

**Pattern**: Thread-safe, async-safe event queue

```python
_stream_queue: ContextVar[Queue | None] = ContextVar("_stream_queue", default=None)

async def astream(self, **inputs):
    queue = asyncio.Queue()
    token = _stream_queue.set(queue)
    try:
        # Execute and yield from queue
        async for event in self._yield_from_queue(queue):
            yield event
    finally:
        _stream_queue.reset(token)
```

**Why**: Works across async tasks, no global state

---

### 4. Context-Based Configuration

**Pattern**: Thread-safe configuration overrides

```python
# Global
udspy.settings.configure(model="gpt-4o-mini")

# Context-specific
with udspy.settings.context(model="gpt-4"):
    result = predictor(...)  # Uses gpt-4

# Back to global
result = predictor(...)  # Uses gpt-4o-mini
```

**Why**: Multi-tenant safe, no parameter drilling

---

### 5. Exception-Based Flow Control

**Pattern**: Use exceptions for suspend/resume

```python
try:
    result = agent(question="...")
except ConfirmationRequired as e:
    response = input(e.question)
    result = agent(resume_state=ResumeState(e, response))
```

**Why**: Clean interrupt, preserves state, composable

---

## Decision Tree

### Where should this code go?

```
Is it about LLM provider API calls?
├─ YES → LM Layer (future: src/udspy/lm/)
└─ NO
   ├─ Is it about message formatting or parsing?
   │  └─ YES → Adapter Layer (src/udspy/adapter.py)
   └─ NO
      ├─ Is it about business logic or orchestration?
      │  └─ YES → Module Layer (src/udspy/module/)
      └─ NO
         ├─ Is it about conversation storage?
         │  └─ YES → History (src/udspy/history.py)
         ├─ Is it about tool definition or execution?
         │  └─ YES → Tool (src/udspy/tool.py)
         ├─ Is it about streaming events?
         │  └─ YES → Streaming (src/udspy/streaming.py)
         ├─ Is it about human-in-the-loop?
         │  └─ YES → Confirmation (src/udspy/confirmation.py)
         ├─ Is it about telemetry?
         │  └─ YES → Callbacks (src/udspy/callback.py)
         └─ Is it a utility used everywhere?
            └─ YES → Utils (src/udspy/utils.py)
```

### Should this be a new module?

```
Does it encapsulate LLM call logic?
├─ NO → Not a module (maybe a helper/utility)
└─ YES
   ├─ Is it a variant of Predict?
   │  ├─ YES → Probably wrapper (like ChainOfThought)
   │  └─ NO → Custom module
   └─ Does it need custom orchestration?
      ├─ YES → Create new module (like ReAct)
      └─ NO → Compose existing modules
```

### Should this be in the LM layer?

```
Does it talk to an LLM provider API?
├─ NO → Not LM layer
└─ YES
   ├─ Is it provider-specific (OpenAI, Anthropic)?
   │  └─ YES → LM implementation (src/udspy/lm/openai.py)
   └─ Is it provider-agnostic (retry, rate limiting)?
      └─ YES → LM base (src/udspy/lm/base.py)
```

---

## Summary

### Responsibilities at a Glance

| Layer | Responsibilities | Key Files |
|-------|-----------------|-----------|
| **Module** | Business logic, orchestration, composition | `module/predict.py`, `module/react.py` |
| **LM** | Provider API calls, retries, format conversion | *(Future)* `lm/openai.py`, `lm/anthropic.py` |
| **Adapter** | Message formatting, output parsing, schema conversion | `adapter.py` |
| **Signature** | I/O contracts, validation | `signature.py` |
| **Tool** | Function wrapping, schema extraction, execution | `tool.py` |
| **History** | Conversation storage | `history.py` |
| **Streaming** | Event queue, chunks, emission | `streaming.py` |
| **Confirmation** | Suspend/resume, human-in-the-loop | `confirmation.py` |
| **Callbacks** | Telemetry, monitoring hooks | `callback.py` |
| **Settings** | Configuration, client management | `settings.py` |

### Core Patterns

1. **Async-first** - All core operations are async
2. **Single execution path** - `aexecute()` powers everything
3. **Event queue** - ContextVar for streaming
4. **Exception-based flow** - ConfirmationRequired for suspend/resume
5. **Context-based config** - Thread-safe overrides

### Next Steps

For implementation details, see:
- [Modules Deep Dive](modules.md)
- [Signatures Deep Dive](signatures.md)
- [Streaming Deep Dive](streaming.md)
- [Confirmation Deep Dive](confirmation.md)
- [Callbacks Deep Dive](callbacks.md)
- [Architectural Decisions](decisions.md)
