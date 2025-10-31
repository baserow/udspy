# Architectural Changes

This document tracks major architectural decisions and changes made to udspy.

## 2025-10-24: Initial Project Setup

### Context
Created a minimal DSPy-inspired library focused on:
- Simplicity over feature completeness
- Native OpenAI tool calling instead of custom adapters
- Streaming support for reasoning and output fields
- Modern Python tooling (uv, ruff, justfile)

### Key Design Decisions

1. **Native Tool Calling**
   - Unlike DSPy which uses custom adapters and field markers, udspy uses OpenAI's native function calling API
   - Tools are defined as Pydantic models and automatically converted to OpenAI tool schemas
   - This reduces complexity and leverages OpenAI's optimized tool calling

2. **Minimal Dependencies**
   - Only `openai` and `pydantic` in core dependencies
   - Keeps the library lightweight and maintainable
   - Reduces potential dependency conflicts

3. **Pydantic v2**
   - Uses Pydantic v2 for all models and validation
   - Leverages new features like model_json_schema() for tool definitions
   - Better performance and more modern API

4. **Streaming Architecture**
   - Async-first design using Python's async/await
   - Separate field streaming for reasoning and outputs
   - Field boundaries detected using simple delimiters or JSON parsing

5. **Module Abstraction**
   - Similar to DSPy but simplified
   - Modules compose via Python class inheritance
   - Signatures define I/O contracts using Pydantic models
   - Predict is the core primitive for LLM calls

### Project Structure

```
udspy/
├── src/udspy/           # Core library code
│   ├── signature.py     # Signature, InputField, OutputField
│   ├── adapter.py       # ChatAdapter for formatting
│   ├── module.py        # Module, Predict abstractions
│   └── streaming.py     # Streaming support
├── tests/               # Pytest tests
├── docs/                # MkDocs documentation
├── examples/            # Usage examples
├── pyproject.toml       # Project config (uv-based)
├── justfile            # Command runner
└── .github/workflows/   # CI configuration
```

### Future Considerations

1. **Adapter Extensibility**: May need to support other LLM providers (Anthropic, etc.)
2. **Advanced Streaming**: Consider field-specific callbacks or transformations
3. **Optimization**: Room for prompt optimization and few-shot learning like DSPy
4. **Tool Execution**: May add built-in tool executor with retry logic

---

## 2025-10-24: Context Manager for Settings

### Context
Need to support different API keys and models in different contexts (e.g., multi-tenant apps, different users, testing scenarios).

### Decision
Implemented thread-safe context manager using Python's `contextvars` module:

```python
from udspy.lm import LM

# Global settings
global_lm = LM(model="gpt-4o-mini", api_key="global-key")
udspy.settings.configure(lm=global_lm)

# Temporary override in context
user_lm = LM(model="gpt-4", api_key="user-key")
with udspy.settings.context(lm=user_lm):
    result = predictor(question="...")  # Uses user-key and gpt-4

# Back to global settings
result = predictor(question="...")  # Uses global-key and gpt-4o-mini
```

### Key Features

1. **Thread-Safe**: Uses `ContextVar` for thread-safe context isolation
2. **Nestable**: Contexts can be nested with proper inheritance
3. **Comprehensive**: Supports overriding lm, callbacks, and any kwargs
4. **Clean API**: Simple context manager interface with LM instances
5. **Flexible**: Use different LM providers per context

### Implementation Details

- Added `ContextVar` fields to `Settings` class for each configurable attribute
- Properties now check context first, then fall back to global settings
- Context manager saves/restores context state using try/finally
- Proper cleanup ensures no context leakage

### Use Cases

1. **Multi-tenant applications**: Different API keys per user
   ```python
   user_lm = LM(model="gpt-4o-mini", api_key=user.api_key)
   with udspy.settings.context(lm=user_lm):
       result = predictor(question=user.question)
   ```

2. **Model selection per request**: Use different models for different tasks
   ```python
   powerful_lm = LM(model="gpt-4", api_key=api_key)
   with udspy.settings.context(lm=powerful_lm):
       result = expensive_predictor(question=complex_question)
   ```

3. **Testing**: Isolate test settings without affecting global state
   ```python
   test_lm = LM(model="gpt-4o-mini", api_key="sk-test")
   with udspy.settings.context(lm=test_lm, temperature=0.0):
       assert predictor(question="2+2").answer == "4"
   ```

4. **Async operations**: Safe concurrent operations with different settings
   ```python
   async def handle_user(user):
       user_lm = LM(model="gpt-4o-mini", api_key=user.api_key)
       with udspy.settings.context(lm=user_lm):
           async for chunk in streaming_predictor.stream(...):
               yield chunk
   ```

### Consequences

**Benefits**:
- Clean separation of concerns (global vs context-specific settings)
- No need to pass settings through function parameters
- Thread-safe for concurrent operations
- Flexible and composable

**Trade-offs**:
- Slight complexity increase in Settings class
- Context variables have a small performance overhead (negligible)
- Must remember to use context manager (but gracefully degrades to global settings)

### Migration Guide
No migration needed - feature is additive and backwards compatible.

---

## 2025-10-24: Chain of Thought Module

### Context
Chain of Thought (CoT) is a proven prompting technique that improves LLM reasoning by explicitly requesting step-by-step thinking. This is one of the most valuable patterns from DSPy.

### Decision
Implemented `ChainOfThought` module that automatically adds a reasoning field to any signature:

```python
class QA(Signature):
    """Answer questions."""
    question: str = InputField()
    answer: str = OutputField()

# Automatically extends to: question -> reasoning, answer
cot = ChainOfThought(QA)
result = cot(question="What is 15 * 23?")

print(result.reasoning)  # Shows step-by-step calculation
print(result.answer)     # "345"
```

### Implementation Approach

Unlike DSPy which uses a `signature.prepend()` method, udspy takes a simpler approach:

1. **Extract fields** from original signature
2. **Create extended outputs** with reasoning prepended: `{"reasoning": str, **original_outputs}`
3. **Use make_signature** to create new signature dynamically
4. **Wrap in Predict** with the extended signature

This approach:
- Doesn't require adding prepend/insert methods to Signature
- Leverages existing `make_signature` utility
- Keeps ChainOfThought as a pure Module wrapper
- Only ~45 lines of code

### Key Features

1. **Automatic reasoning field**: No manual signature modification needed
2. **Customizable description**: Override reasoning field description
3. **Works with any signature**: Single or multiple outputs
4. **Transparent**: Reasoning is always accessible in results
5. **Configurable**: All Predict parameters (model, temperature, tools) supported

### Research Evidence

Chain of Thought prompting improves performance on:
- **Math**: ~25-30% accuracy improvement (Wei et al., 2022)
- **Reasoning**: Significant gains on logic puzzles
- **Multi-step**: Better at complex multi-hop reasoning
- **Transparency**: Shows reasoning for verification

### Use Cases

1. **Math and calculation**
   ```python
   cot = ChainOfThought(QA, temperature=0.0)
   result = cot(question="What is 157 * 234?")
   ```

2. **Analysis and decision-making**
   ```python
   class Decision(Signature):
       scenario: str = InputField()
       decision: str = OutputField()
       justification: str = OutputField()

   decider = ChainOfThought(Decision)
   ```

3. **Educational applications**: Show work/reasoning
4. **High-stakes decisions**: Require explicit justification
5. **Debugging**: Understand why LLM made specific choices

### Consequences

**Benefits**:
- Improved accuracy on reasoning tasks
- Transparent reasoning process
- Easy to verify correctness
- Simple API (just wrap any signature)
- Minimal code overhead

**Trade-offs**:
- Increased token usage (~2-3x for simple tasks)
- Slightly higher latency
- Not always needed for simple factual queries
- Reasoning quality depends on model capability

### Comparison with DSPy

| Aspect | udspy | DSPy |
|--------|-------|------|
| API | `ChainOfThought(signature)` | `dspy.ChainOfThought(signature, rationale_field=...)` |
| Implementation | Dynamic signature creation | Signature.prepend() method |
| Customization | `reasoning_description` param | Full `rationale_field` control |
| Complexity | ~45 lines | ~40 lines |
| Dependencies | Uses `make_signature` | Uses signature mutation |

Both are equally effective; udspy's approach is simpler but less flexible in edge cases.

### Future Considerations

1. **Streaming support**: StreamingChainOfThought for incremental reasoning
2. **Few-shot examples**: Add example reasoning patterns to improve quality
3. **Verification**: Automatic reasoning quality checks
4. **Caching**: Built-in caching for repeated queries

### Migration Guide
Feature is additive - no migration needed.

---

## 2025-10-31: Module Callbacks and Dynamic Tool Management

### Context
Need to support dynamic tool loading where tools can modify the available toolset during execution. Use cases include:
- Loading specialized tools only when needed (performance, cost)
- Progressive tool discovery (agent figures out what it needs)
- Category-based tool loading (math, web, data tools)
- Multi-tenant tool sets (user-specific permissions)

### Decision
Implemented module callback system allowing tools to return special callables that modify module state:

```python
@tool(name="load_calculator", description="Load calculator tool")
def load_calculator() -> callable:
    """Load calculator tool dynamically."""

    @module_callback
    def add_calculator(context):
        # Get current tools
        current_tools = [
            t for t in context.module.tools.values()
            if t.name not in ("finish", "ask_to_user")
        ]

        # Add calculator to available tools
        context.module.init_module(tools=current_tools + [calculator])

        return "Calculator loaded successfully"

    return add_calculator

# Agent starts with only the loader
agent = ReAct(Question, tools=[load_calculator])

# Agent loads calculator when needed, then uses it
result = agent(question="What is 157 * 834?")
```

### Key Features

1. **@module_callback Decorator**: Marks callables as module callbacks
2. **Context Objects**: Pass execution context to callbacks
   - `ReactContext`: Includes trajectory history
   - `PredictContext`: Includes conversation history
   - `ModuleContext`: Base context with module reference
3. **init_module() Pattern**: Unified method to reinitialize tools and signatures
4. **Tool Persistence**: Dynamically loaded tools persist for entire execution
5. **Observation Return**: Callbacks must return string for trajectory

### Implementation Approach

1. **Decorator-based marking**: Simple `@module_callback` decorator to mark callables
2. **Return value detection**: Check if tool result is a module callback
3. **Context injection**: Pass appropriate context object to callback
4. **Module modification**: Callbacks use `context.module.init_module(tools=[...])`
5. **Signature regeneration**: init_module() rebuilds both tools and signatures

This approach:
- Keeps tool functions simple (just return the callback)
- Provides full module access through context
- Works with all module types (Predict, ChainOfThought, ReAct)
- Maintains clean separation of concerns

### Consequences

**Benefits**:
- On-demand tool loading reduces token usage and context size
- Progressive discovery allows adaptive agent behavior
- Clean API with decorator pattern
- Full module state access through context
- Works seamlessly with existing tool system

**Trade-offs**:
- Additional complexity in tool execution logic
- Must remember to return string from callbacks
- Tool persistence requires new instance for fresh state
- Context objects add memory overhead (minimal)

**Alternatives Considered**:
- **Direct module mutation**: Rejected due to lack of encapsulation
- **Event system**: Rejected as too complex for the use case
- **Plugin architecture**: Rejected as overkill for tool management

### Use Cases

1. **On-demand capabilities**: Load expensive tools only when needed
2. **Progressive discovery**: Agent discovers tools as it works
3. **Multi-tenant**: Load user-specific tools based on permissions
4. **Adaptive tool sets**: Adjust tools based on task complexity
5. **Category loading**: Load tool groups (math, web, data)

### Migration Guide
Feature is additive - existing code continues to work. To use dynamic tools:
1. Define tools that return `@module_callback` decorated functions
2. Callbacks receive context and call `context.module.init_module(tools=[...])`
3. Return string observation from callback

---

## 2025-10-31: Confirmation System (Human-in-the-Loop)

### Context
Need to support human approval before executing sensitive operations. Use cases include:
- Dangerous tool calls (delete files, send emails, make purchases)
- High-stakes decisions requiring human judgment
- User editing of tool arguments before execution
- Interactive workflows with human feedback

### Decision
Implemented confirmation system using exceptions for control flow:

```python
@confirm_first
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

# First call - raises ConfirmationRequired
try:
    delete_file("/tmp/important.txt")
except ConfirmationRequired as e:
    print(f"Confirm: {e.question}")
    # User approves
    respond_to_confirmation(e.confirmation_id, approved=True)

# Retry - proceeds with execution
result = delete_file("/tmp/important.txt")  # Actually deletes file
```

### Key Features

1. **@confirm_first Decorator**: Wraps any function to require confirmation
2. **Exception-Based Control**: Uses ConfirmationRequired/ConfirmationRejected exceptions
3. **Stable IDs**: Hash-based confirmation IDs from function name + arguments
4. **Argument Editing**: Users can modify arguments before execution
5. **Context Storage**: Thread-safe ContextVar for approval state
6. **Status Tracking**: Literal types for type-safe status ("pending", "approved", "rejected", "edited", "feedback")

### Implementation Approach

1. **Exception for control flow**: ConfirmationRequired pauses execution
2. **Context-based approval**: Store approvals in ContextVar dict
3. **Stable confirmation IDs**: Hash(function_name + args) for resumption
4. **Three-state flow**:
   - **Pending**: No approval → raises ConfirmationRequired
   - **Approved**: Has approval → proceeds with execution
   - **Rejected**: User rejected → raises ConfirmationRejected
5. **Argument modification**: Pass modified args via `data` parameter

This approach:
- Clean separation of concerns (decorator handles confirmation logic)
- Exception-based control flow is Pythonic
- Stable IDs allow resumption after approval
- Thread-safe with ContextVar
- Works with sync and async functions

### Key Types

```python
# Type-safe status
ConfirmationStatus = Literal["pending", "approved", "rejected", "edited", "feedback"]

# Typed approval data
class ApprovalData(TypedDict, total=False):
    approved: bool
    data: dict[str, Any] | None
    status: ConfirmationStatus

# Exception classes
class ConfirmationRequired(Exception):
    """Raised when confirmation is needed"""
    question: str
    confirmation_id: str
    tool_call: ToolCall | None
    context: dict[str, Any]

class ConfirmationRejected(Exception):
    """Raised when user rejects operation"""
    message: str
    confirmation_id: str
    tool_call: ToolCall | None
```

### Integration with ReAct

ReAct module automatically catches ConfirmationRequired and adds execution state to context:

```python
try:
    result = await tool.acall(**tool_args)
except ConfirmationRequired as e:
    # Add ReAct state to exception context
    e.context = {
        "trajectory": trajectory.copy(),
        "iteration": idx,
        "input_args": input_args.copy(),
    }
    if e.tool_call and tool_call_id:
        e.tool_call.call_id = tool_call_id
    raise  # Re-raise for caller to handle
```

This allows resuming ReAct execution from the exact point of interruption.

### Consequences

**Benefits**:
- Safe execution of dangerous operations
- User can edit arguments before execution
- Clean exception-based control flow
- Thread-safe with ContextVar
- Type-safe with Literal types and TypedDict
- Integrates cleanly with existing code

**Trade-offs**:
- Exception-based control flow can be surprising
- Must remember to handle ConfirmationRequired
- Confirmation state is per-process (doesn't persist across restarts)
- Hash-based IDs could collide (extremely rare)

**Alternatives Considered**:
- **Callback-based**: Rejected as less Pythonic than exceptions
- **Async/await pattern**: Rejected due to complexity with mixed sync/async
- **Global registry**: Rejected due to testing difficulties

### Use Cases

1. **Dangerous operations**: File deletion, database changes, API calls
2. **User editing**: Modify tool arguments before execution
3. **High-stakes decisions**: Require explicit approval with reasoning
4. **Interactive workflows**: Back-and-forth with human feedback
5. **Testing**: Mock confirmations in test suite

### Migration Guide
Feature is additive - existing code continues to work. To use confirmations:
1. Decorate functions with `@confirm_first`
2. Catch `ConfirmationRequired` exception
3. Call `respond_to_confirmation(id, approved=True/False)`
4. Retry the function call (uses approval from context)
5. Handle `ConfirmationRejected` if user rejects

---

## 2025-10-31: LM Callable Interface with String Prompts

### Context
Users want the simplest possible interface for quick LLM queries without needing to construct message lists. Common use case is prototyping and simple scripts where full message structure is overkill.

### Decision
Enhanced LM base class to accept simple string prompts and return just the text:

```python
from udspy import OpenAILM
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")
lm = OpenAILM(client=client, default_model="gpt-4o-mini")

# Simple string prompt - returns just text
answer = lm("What is the capital of France?")
print(answer)  # "Paris"

# Override model
answer = lm("Explain quantum physics", model="gpt-4")

# With parameters
answer = lm("Write a haiku", temperature=0.9, max_tokens=100)
```

### Implementation Approach

1. **Overloaded `__call__` method**: Two signatures via `@overload`
   - `str` → returns `str` (text only)
   - `list[dict[str, Any]]` → returns `Any` (full response)

2. **Automatic message wrapping**: String prompts wrapped as `[{"role": "user", "content": prompt}]`

3. **Response extraction**: For string prompts, extract just the text content from `response.choices[0].message.content`

4. **Optional model parameter**: Made `model` parameter optional everywhere, falls back to `self.model` property

5. **Type hints**: Proper overloads for IDE autocomplete and type checking

This approach:
- Maintains backward compatibility (existing code works unchanged)
- Provides ergonomic interface for simple cases
- Preserves full power for complex cases
- Clear type hints for IDE support
- Falls back gracefully if text extraction fails

### Key Changes

```python
class LM(ABC):
    @property
    def model(self) -> str | None:
        """Get default model for this LM instance."""
        return None

    @overload
    def __call__(self, prompt: str, *, model: str | None = None, **kwargs: Any) -> str: ...

    @overload
    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any: ...

    def __call__(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str | Any:
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
            response = self.complete(messages, model=model, **kwargs)
            # Extract just the text content
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, "content") and message.content:
                    return message.content
            return str(response)
        else:
            return self.complete(prompt_or_messages, model=model, **kwargs)
```

### Consequences

**Benefits**:
- Simplest possible API for quick queries
- No need to construct message dictionaries for simple cases
- Backward compatible with existing code
- Proper type hints for IDE support
- Falls back gracefully if needed
- Model parameter now optional everywhere

**Trade-offs**:
- Slight complexity in `__call__` implementation
- String/list dispatch adds minor overhead (negligible)
- Text extraction logic specific to OpenAI response format
- Two different return types require overloads for type safety

**Alternatives Considered**:
- **Separate method** (`lm.ask()`): Rejected as less convenient
- **Always return text**: Rejected as losing access to full response
- **Factory function**: Rejected as less object-oriented

### Use Cases

1. **Prototyping**: Quick tests without boilerplate
2. **Simple scripts**: One-line LLM queries
3. **Interactive sessions**: REPL-friendly API
4. **Learning**: Easiest API for newcomers
5. **Utilities**: Simple helper functions

### Migration Guide
No migration needed - feature is additive and backward compatible. New usage pattern:
```python
# Before: Required message construction
response = lm.complete([{"role": "user", "content": "Hello"}], model="gpt-4o")
text = response.choices[0].message.content

# After: Direct string prompt
text = lm("Hello", model="gpt-4o")
```

---

## 2025-10-31: History Management with System Prompts

### Context
Chat histories need special handling for system prompts to ensure they're always first. Module behavior depends on having system instructions properly positioned, and tools may manipulate histories during execution.

### Decision
Implemented `History` class with dedicated system prompt management:

```python
history = History()

# System prompt always stays first, even if added later
history.add_message(role="user", content="Hello")
history.add_message(role="assistant", content="Hi!")
history.system_prompt = "You are a helpful assistant"  # Goes to front

messages = history.messages
# [{"role": "system", "content": "You are a helpful assistant"},
#  {"role": "user", "content": "Hello"},
#  {"role": "assistant", "content": "Hi!"}]
```

### Key Features

1. **Dedicated system_prompt property**: Special handling for system messages
2. **Automatic positioning**: System prompt always first in messages list
3. **Mutable**: Can update system prompt at any time, position is maintained
4. **Copy support**: history.copy() includes system prompt
5. **Clear separation**: Other messages in separate list from system prompt

### Implementation Approach

```python
class History:
    def __init__(self, system_prompt: str | None = None):
        self._messages: list[dict[str, Any]] = []
        self._system_prompt: str | None = system_prompt

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Get all messages with system prompt first (if set)."""
        if self._system_prompt:
            return [
                {"role": "system", "content": self._system_prompt},
                *self._messages
            ]
        return self._messages.copy()
```

This approach:
- System prompt stored separately from regular messages
- `messages` property dynamically constructs full list
- No risk of system prompt appearing mid-conversation
- Simple to update system prompt without rebuilding list
- Clear ownership (History manages system message)

### Consequences

**Benefits**:
- System prompt guaranteed to be first
- Can update system prompt at any time
- Clean API with property access
- Prevents common mistakes (system prompt mid-conversation)
- Supports all history manipulation patterns

**Trade-offs**:
- Small overhead constructing messages list each access
- System message can't be treated like regular message
- Slight complexity in History implementation

**Alternatives Considered**:
- **Insert at index 0**: Rejected as error-prone with mutations
- **Validation on add**: Rejected as too restrictive
- **Separate system field**: Chosen approach

### Use Cases

1. **Module initialization**: Set system prompt per module type
2. **Dynamic prompts**: Update based on context or user
3. **Tool manipulation**: Tools can update system prompt safely
4. **History replay**: Maintain system prompt across sessions
5. **Multi-turn conversations**: System prompt persists correctly

### Migration Guide
Existing code using History.add_message() continues to work. To use system prompts:
```python
# Create with system prompt
history = History(system_prompt="You are helpful")

# Or set later
history.system_prompt = "You are a math tutor"

# Always correctly positioned in messages
messages = history.messages  # System prompt is first
```

---

## Template for Future Entries

## YYYY-MM-DD: Change Title

### Context
Why was this change needed?

### Decision
What was decided and implemented?

### Consequences
- What are the benefits?
- What are the trade-offs?
- What alternatives were considered?

### Migration Guide (if applicable)
How should users update their code?
