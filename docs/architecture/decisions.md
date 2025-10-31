# Architectural Decision Records (ADR)

This document tracks major architectural decisions made in udspy, presented chronologically with context, rationale, and consequences.

## Table of Contents

1. [Initial Project Setup (2025-10-24)](#adr-001-initial-project-setup)
2. [Context Manager for Settings (2025-10-24)](#adr-002-context-manager-for-settings)
3. [Chain of Thought Module (2025-10-24)](#adr-003-chain-of-thought-module)
4. [Human-in-the-Loop with Confirmation System (2025-10-25)](#adr-004-human-in-the-loop-with-confirmation-system)
5. [ReAct Agent Module (2025-10-25)](#adr-005-react-agent-module)
6. [Unified Module Execution Pattern (aexecute) (2025-10-25)](#adr-006-unified-module-execution-pattern-aexecute)
7. [Automatic Retry on Parse Errors (2025-10-29)](#adr-007-automatic-retry-on-parse-errors)
8. [Module Callbacks and Dynamic Tool Management (2025-10-31)](#adr-008-module-callbacks-and-dynamic-tool-management)
9. [History Management with System Prompts (2025-10-31)](#adr-009-history-management-with-system-prompts)
10. [LM Callable Interface with String Prompts (2025-10-31)](#adr-010-lm-callable-interface-with-string-prompts)

---

## ADR-001: Initial Project Setup

**Date**: 2025-10-24

**Status**: Accepted

### Context

Needed a minimal library for LLM-powered applications in resource-constrained environments, specifically for Baserow's AI assistant where ~200MB dependencies are prohibitive.

### Decision

Build a lightweight library with:
- Native OpenAI tool calling as the primary approach
- Minimal dependencies (~10MB: `openai` + `pydantic`)
- Streaming support for reasoning and output fields
- Async-first architecture
- Modern Python tooling (uv, ruff, justfile)

**Note**: Heavily inspired by [DSPy's](https://github.com/stanfordnlp/dspy) excellent abstractions and API patterns.

### Key Design Decisions

#### 1. Native Tool Calling

Use OpenAI's native function calling API directly as the primary approach.

**Rationale**:
- OpenAI's tool calling is optimized and well-tested
- Reduces complexity - no need for multi-provider adapter layer
- Forward compatible with future OpenAI improvements
- Works with any OpenAI-compatible provider (Together, Ollama, etc.)
- Sufficient for Baserow's AI assistant needs

**Trade-offs**:
- Couples to OpenAI's API format (acceptable for our use case)
- Limited to OpenAI-compatible providers

#### 2. Minimal Dependencies

Only `openai` and `pydantic` in core dependencies.

**Rationale**:
- Keeps the library lightweight (~10MB)
- Reduces potential dependency conflicts in Baserow
- Faster installation and lower memory usage
- Suitable for serverless, edge, and embedded deployments

**Trade-offs**:
- Limited to OpenAI-compatible providers
- No multi-provider abstraction layer

#### 3. Pydantic v2

Use Pydantic v2 for all models and validation.

**Rationale**:
- Modern, fast, well-maintained
- Excellent JSON schema generation for tools
- Built-in validation and type coercion
- Great developer experience with IDE support

**Trade-offs**:
- Requires Python 3.7+ (we target 3.11+)

#### 4. Streaming Architecture

Async-first design using Python's async/await.

**Rationale**:
- Python's async is the standard for I/O-bound operations
- Native support from OpenAI SDK
- Composable with Baserow's async infrastructure
- First-class support for streaming reasoning and outputs

**Trade-offs**:
- Requires async runtime (asyncio)
- Steeper learning curve for beginners

#### 5. Module Abstraction

Modules compose via Python class inheritance.

**Rationale**:
- Familiar Python patterns (no custom DSL)
- Good IDE and type checker support
- Signatures define I/O contracts using Pydantic models
- Predict is the core primitive for LLM calls

**Trade-offs**:
- Requires more explicit code vs meta-programming approaches
- Less abstraction = more boilerplate for advanced use cases

### Consequences

**Benefits**:
- Small memory footprint (~10MB)
- Works in resource-constrained environments (Baserow AI assistant)
- Simple, maintainable codebase
- Compatible with any OpenAI-compatible provider
- Fast installation and startup

**Trade-offs**:
- Limited to OpenAI-compatible providers
- No built-in optimizers or teleprompters
- Fewer abstractions = more manual work for complex scenarios

### Alternatives Considered

- **Use existing frameworks**: Larger footprints, more dependencies
- **Build from scratch**: Chose this - start minimal, add what's needed

---

## ADR-002: Context Manager for Settings

**Date**: 2025-10-24

**Status**: Accepted

### Context

Need to support different API keys and models in different contexts (e.g., multi-tenant apps, different users, testing scenarios, concurrent async operations).

### Decision

Implement thread-safe context manager using Python's `contextvars` module:

```python
from udspy import LM

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

### Implementation Details

- Added `ContextVar` fields to `Settings` class for each configurable attribute
- Properties now check context first, then fall back to global settings
- Context manager saves/restores context state using try/finally
- Proper cleanup ensures no context leakage

### Key Features

1. **Thread-Safe**: Uses `ContextVar` for thread-safe context isolation
2. **Nestable**: Contexts can be nested with proper inheritance
3. **Comprehensive**: Supports overriding lm, callbacks, and any kwargs
4. **Clean API**: Simple context manager interface with LM instances
5. **Flexible**: Use different LM providers per context

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
- Thread-safe and asyncio task-safe for concurrent operations
- Flexible and composable

**Trade-offs**:
- Slight complexity increase in Settings class
- Context variables have a small performance overhead (negligible)
- Must remember to use context manager (but gracefully degrades to global settings)

### Alternatives Considered

- **Dependency Injection**: More verbose, harder to use
- **Environment Variables**: Not dynamic enough for multi-tenant use cases
- **Pass settings everywhere**: Too cumbersome

### Migration Guide

No migration needed - feature is additive and backwards compatible.

---

## ADR-003: Chain of Thought Module

**Date**: 2025-10-24

**Status**: Accepted

### Context

Chain of Thought (CoT) is a proven prompting technique that improves LLM reasoning by explicitly requesting step-by-step thinking. Research shows ~25-30% accuracy improvement on math and reasoning tasks (Wei et al., 2022).

### Decision

Implement `ChainOfThought` module that automatically adds a reasoning field to any signature:

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

### Alternatives Considered

- **Prompt Engineering**: Less reliable than structured reasoning field
- **Tool-based Reasoning**: Too heavyweight for simple reasoning
- **Custom Signature per Use**: Too much boilerplate

### Future Considerations

1. **Streaming support**: StreamingChainOfThought for incremental reasoning
2. **Few-shot examples**: Add example reasoning patterns to improve quality
3. **Verification**: Automatic reasoning quality checks
4. **Caching**: Built-in caching for repeated queries

### Migration Guide

Feature is additive - no migration needed.

---

## ADR-004: Human-in-the-Loop with Confirmation System

**Date**: 2025-10-25 (Updated: 2025-10-31)

**Status**: Accepted

### Context

Many agent applications require human approval for certain actions (e.g., deleting files, sending emails, making purchases). We needed a clean way to suspend execution, ask for user input, and resume where we left off. The system must support:
- Multiple confirmation rounds (clarifications, edits, iterations)
- State preservation for resumption
- Thread-safe concurrent operations
- Integration with ReAct agent trajectories

### Decision

Implement exception-based confirmation system with:
- **Exceptions for control flow**: `ConfirmationRequired`, `ConfirmationRejected`
- **@confirm_first decorator**: Wraps functions to require confirmation
- **ResumeState**: Container for resuming execution after confirmation
- **Type-safe status tracking**: Literal types for compile-time validation
- **Thread-safe context**: Uses `contextvars` for isolated state

```python
from udspy import (
    confirm_first,
    ConfirmationRequired,
    ConfirmationRejected,
    ResumeState,
    respond_to_confirmation
)

@confirm_first
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

# Interactive loop pattern
resume_state = None
while True:
    try:
        result = delete_file("/important.txt", resume_state=resume_state)
        break
    except ConfirmationRequired as e:
        response = input(f"{e.question} (yes/no): ")
        resume_state = ResumeState(e, response)
    except ConfirmationRejected as e:
        print(f"Rejected: {e.message}")
        break
```

### Implementation Details

1. **Stable Confirmation IDs**: Generated from `function_name:hash(args)` for idempotent resumption
2. **Type-safe Status**: `ConfirmationStatus = Literal["pending", "approved", "rejected", "edited", "feedback"]`
3. **ApprovalData TypedDict**: Structured approval data with type safety
4. **ResumeState Container**: Combines exception + user response for clean resumption API
5. **Context Storage**: Thread-safe `ContextVar[dict[str, ApprovalData]]`
6. **Tool Integration**: `check_tool_confirmation()` for tool-level confirmations
7. **Automatic Cleanup**: Confirmations cleared after successful execution

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
    question: str
    confirmation_id: str
    tool_call: ToolCall | None
    context: dict[str, Any]  # Module state for resumption

class ConfirmationRejected(Exception):
    message: str
    confirmation_id: str
    tool_call: ToolCall | None

# Resume state container
class ResumeState:
    exception: ConfirmationRequired
    user_response: str
    confirmation_id: str  # Property
    question: str  # Property
    tool_call: ToolCall | None  # Property
    context: dict[str, Any]  # Property
```

### Resumption Patterns

**Pattern 1: Explicit respond_to_confirmation()**
```python
try:
    delete_file("/data")
except ConfirmationRequired as e:
    respond_to_confirmation(e.confirmation_id, approved=True)
    delete_file("/data")  # Resumes
```

**Pattern 2: ResumeState loop (recommended)**
```python
resume_state = None
while True:
    try:
        result = agent(question="Task", resume_state=resume_state)
        break
    except ConfirmationRequired as e:
        response = get_user_input(e.question)
        resume_state = ResumeState(e, response)
```

### ReAct Integration

ReAct automatically catches `ConfirmationRequired` and adds execution state:

```python
try:
    result = await tool.acall(**tool_args)
except ConfirmationRequired as e:
    # ReAct enriches context with trajectory state
    e.context = {
        "trajectory": trajectory.copy(),
        "iteration": idx,
        "input_args": input_args.copy(),
    }
    if e.tool_call and tool_call_id:
        e.tool_call.call_id = tool_call_id
    raise  # Re-raise for caller
```

This enables resuming from exact point in trajectory.

### Key Features

1. **Exception-based control**: Natural suspension of call stack
2. **ResumeState container**: Clean API for resumption with user response
3. **Type-safe**: Literal types and TypedDict for status tracking
4. **Thread-safe**: `ContextVar` isolation per thread/task
5. **Async-safe**: Works with asyncio concurrent operations
6. **Module integration**: Modules can save/restore state in exception context
7. **Tool confirmations**: `check_tool_confirmation()` for tool-level checks
8. **Argument editing**: Users can modify arguments before approval

### Use Cases

1. **Dangerous operations**: File deletion, system commands, database changes
2. **User confirmation**: Sending emails, making purchases, API calls
3. **Clarification loops**: Ask user for additional information
4. **Argument editing**: Let user modify parameters before execution
5. **Multi-step workflows**: Multiple confirmation rounds in agent execution
6. **Web APIs**: Save state in session, resume later
7. **Batch processing**: Auto-approve low-risk, human review high-risk

### Consequences

**Benefits**:
- Clean separation of business logic from approval logic
- Works naturally with ReAct agent trajectories
- Thread-safe and async-safe out of the box
- Easy to test (deterministic based on confirmation state)
- Type-safe with Literal types and TypedDict
- ResumeState provides clean resumption API
- Supports multiple confirmation rounds
- State preservation enables complex workflows

**Trade-offs**:
- Requires exception handling (explicit and clear)
- Confirmation state is per-process (doesn't persist across restarts)
- Hash-based IDs could collide (extremely rare)
- Learning curve for exception-based control flow
- Must manage confirmation rounds to prevent infinite loops

### Alternatives Considered

- **Callback-based**: More complex, harder to reason about flow
- **Async/await pattern**: Breaks with mixed sync/async code
- **Return sentinel values**: Ambiguous, requires checking every return
- **Async generators with yield**: Breaks module composability
- **Middleware pattern**: Too heavyweight for this use case
- **Global registry**: Testing difficulties, not thread-safe
- **Manual state management**: Error-prone, inconsistent

### Migration Guide

Feature is additive - no migration needed.

**Basic usage**:
```python
@confirm_first
def dangerous_op():
    ...
```

**Recommended pattern**:
```python
from udspy import ResumeState

resume_state = None
while True:
    try:
        result = agent(question="...", resume_state=resume_state)
        break
    except ConfirmationRequired as e:
        response = input(f"{e.question}: ")
        resume_state = ResumeState(e, response)
```

### See Also

- [Confirmation Architecture](confirmation.md) - Detailed architecture and patterns
- [Confirmation API](../api/confirmation.md) - API documentation
- [ReAct Module](modules/react.md) - Integration with agents

---

## ADR-005: ReAct Agent Module

**Date**: 2025-10-25

**Status**: Accepted

### Context

The ReAct (Reasoning + Acting) pattern combines chain-of-thought reasoning with tool usage in an iterative loop. This is essential for building agents that can solve complex tasks by breaking them down and using tools.

### Decision

Implement a `ReAct` module that:
- Alternates between reasoning and tool execution
- Supports human-in-the-loop for clarifications and confirmations
- Tracks full trajectory of reasoning and actions
- Handles errors gracefully with retries
- Works with both streaming and non-streaming modes

```python
from udspy import ReAct, InputField, OutputField, Signature, tool

@tool(name="search")
def search(query: str) -> str:
    return search_api(query)

class ResearchTask(Signature):
    """Research and answer questions."""
    question: str = InputField()
    answer: str = OutputField()

agent = ReAct(ResearchTask, tools=[search], max_iters=5)
result = agent(question="What is the population of Tokyo?")
```

### Implementation Approach

1. **Iterative Loop**: Continues until final answer or max iterations
2. **Dynamic Signature**: Extends signature with reasoning_N, tool_name_N, tool_args_N fields
3. **Tool Execution**: Automatically executes tools and adds results to context
4. **Error Handling**: Retries with error feedback if tool execution fails
5. **Human Confirmations**: Integrates with `@confirm_first` for user input

### Key Features

1. **Flexible Tool Usage**: Agent decides when and which tools to use
2. **Self-Correction**: Can retry if tool execution fails
3. **Trajectory Tracking**: Full history of reasoning and actions
4. **Streaming Support**: Can stream reasoning in real-time
5. **Human-in-the-Loop**: Built-in support for asking users

### Research Evidence

ReAct improves performance on:
- **Complex Tasks**: 15-30% improvement on multi-step reasoning (Yao et al., 2023)
- **Tool Usage**: More accurate tool selection vs. pure CoT
- **Error Recovery**: Better handling of failed tool calls

### Use Cases

1. **Research Agents**: Answer questions using search and APIs
2. **Task Automation**: Multi-step workflows with tool usage
3. **Data Analysis**: Fetch data, analyze, and summarize
4. **Interactive Assistants**: Ask users for clarification when needed

### Consequences

**Benefits**:
- Powerful agent capabilities with minimal code
- Transparent reasoning process
- Handles complex multi-step tasks
- Built-in error handling and retries

**Trade-offs**:
- Higher token usage due to multiple iterations
- Slower than single-shot predictions
- Quality depends on LLM's reasoning ability
- Can get stuck in loops if not properly configured

### Alternatives Considered

- **Chain-based approach**: Too rigid, hard to add dynamic behavior
- **State machine**: Overly complex for the use case
- **Pure prompting**: Less reliable than structured approach

### Future Considerations

1. **Memory/History**: Long-term memory across sessions
2. **Tool Chaining**: Automatic sequencing of tool calls
3. **Parallel Tool Execution**: Execute independent tools concurrently
4. **Learning**: Optimize tool selection based on feedback

### Migration Guide

Feature is additive - no migration needed.

---

## ADR-006: Unified Module Execution Pattern (aexecute)

**Date**: 2025-10-25

**Status**: Accepted

### Context

Initially, `astream()` and `aforward()` had duplicated logic for executing modules. This made maintenance difficult and increased the chance of bugs when updating behavior.

### Decision

Introduce a single `aexecute()` method that handles both streaming and non-streaming execution:

```python
class Module:
    async def aexecute(self, *, stream: bool = False, **inputs):
        """Core execution logic - handles both streaming and non-streaming."""
        # Implementation here

    async def astream(self, **inputs):
        """Public streaming API."""
        async for event in self.aexecute(stream=True, **inputs):
            yield event

    async def aforward(self, **inputs):
        """Public non-streaming API."""
        async for event in self.aexecute(stream=False, **inputs):
            if isinstance(event, Prediction):
                return event
```

### Implementation Details

1. **Single Source of Truth**: All execution logic in `aexecute()`
2. **Stream Parameter**: Boolean flag controls behavior
3. **Generator Pattern**: Always yields events, even in non-streaming mode
4. **Clean Separation**: Public methods are thin wrappers

### Key Benefits

1. **No Duplication**: Write logic once, use in both modes
2. **Easier Testing**: Test one method instead of two
3. **Consistent Behavior**: Streaming and non-streaming guaranteed to behave identically
4. **Maintainable**: Changes only need to be made in one place
5. **Extensible**: Easy to add new execution modes

### Consequences

**Benefits**:
- Reduced code duplication (~40% less code in modules)
- Easier to maintain and debug
- Consistent behavior across modes
- Simpler to understand (one execution path)

**Trade-offs**:
- Slightly more complex to implement initially
- Need to handle both streaming and non-streaming cases in same method
- Generator pattern requires understanding of async generators

### Before and After

**Before:**
```python
async def astream(self, **inputs):
    # 100 lines of logic
    ...

async def aforward(self, **inputs):
    # 100 lines of DUPLICATED logic with minor differences
    ...
```

**After:**
```python
async def aexecute(self, *, stream: bool, **inputs):
    # 100 lines of logic (used by both)
    ...

async def astream(self, **inputs):
    async for event in self.aexecute(stream=True, **inputs):
        yield event

async def aforward(self, **inputs):
    async for event in self.aexecute(stream=False, **inputs):
        if isinstance(event, Prediction):
            return event
```

### Naming Rationale

We chose `aexecute()` (without underscore prefix) because:
- **Public Method**: This is the main extension point for subclasses
- **Clear Intent**: "Execute" is explicit about what it does
- **Python Conventions**: No underscore = public API, expected to be overridden
- **Not Abbreviated**: Full word avoids ambiguity (vs `aexec` or `acall`)

### Migration Guide

**For Users**: No changes needed - public API remains the same

**For Module Authors**: When creating custom modules, implement `aexecute()` instead of both `astream()` and `aforward()`.

---

## Additional Design Decisions

### Field Markers for Parsing

**Decision**: Use `[[ ## field_name ## ]]` markers to delineate fields in completions.

**Rationale**:
- Simple, regex-parseable format
- Clear visual separation
- Consistent with DSPy's approach (proven)
- Fallback when native tools aren't available

**Trade-offs**:
- Requires careful prompt engineering
- LLM might not always respect markers
- Uses extra tokens

---

## See Also

- [CLAUDE.md](https://github.com/silvestrid/udspy/blob/main/CLAUDE.md) - Chronological architectural changes (development log)
- [Architecture Overview](overview.md) - Component relationships
- [Contributing Guide](https://github.com/silvestrid/udspy/blob/main/CONTRIBUTING.md) - How to propose new decisions

---

## ADR-007: Automatic Retry on Parse Errors

**Date**: 2025-10-29

**Status**: Accepted

### Context

LLMs occasionally generate responses that don't match the expected output format, causing `AdapterParseError` to be raised. This is especially common with:
- Field markers being omitted or malformed
- JSON parsing errors in structured outputs
- Missing required output fields
- Format inconsistencies

These errors are usually transient - the LLM can often generate a valid response on retry. Without automatic retry, users had to implement retry logic themselves, leading to boilerplate code and inconsistent error handling.

### Decision

Implement automatic retry logic using the `tenacity` library on both `Predict._aforward()` and `Predict._astream()` methods:

```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception_type(AdapterParseError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.1, max=3),
)
async def _aforward(self, completion_kwargs: dict[str, Any], should_emit: bool) -> Prediction:
    """Process non-streaming LLM call with automatic retry on parse errors.

    Retries up to 2 times (3 total attempts) with exponential backoff (0.1-3s)
    when AdapterParseError occurs, giving the LLM multiple chances to format
    the response correctly.
    """
```

**Key parameters**:
- **Max attempts**: 3 (1 initial + 2 retries)
- **Retry condition**: Only retry on `AdapterParseError` (not other exceptions)
- **Wait strategy**: Exponential backoff starting at 0.1s, max 3s
- **Applies to**: Both streaming (`_astream`) and non-streaming (`_aforward`) execution

### Implementation Details

1. **Decorator location**: Applied to internal `_aforward` and `_astream` methods (not public API methods)
2. **Tenacity library**: Minimal dependency (~50KB) with excellent async support
3. **Error propagation**: After 3 failed attempts, raises `tenacity.RetryError` wrapping the original `AdapterParseError`
4. **Test isolation**: Tests use a `fast_retry` fixture in `conftest.py` that patches retry decorators to use `wait_none()` for instant retries

### Consequences

**Benefits**:
- **Improved reliability**: Transient parse errors are automatically recovered
- **Better user experience**: Users don't see spurious errors from LLM format issues
- **Reduced boilerplate**: No need for users to implement retry logic
- **Consistent behavior**: All modules get retry logic automatically
- **Configurable backoff**: Exponential backoff prevents API hammering

**Trade-offs**:
- **Increased latency on errors**: Failed attempts add 0.1-3s delay per retry (max ~6s for 3 attempts)
- **Hidden failures**: First 2 parse errors are not visible to users (but logged internally)
- **Token usage**: Failed attempts consume tokens without producing results
- **Test complexity**: Tests need to mock/patch retry behavior to avoid slow tests

### Alternatives Considered

**1. No automatic retry** (status quo before this ADR)
- **Pros**: Simpler, explicit, no hidden behavior
- **Cons**: Every user has to implement retry logic themselves
- **Rejected**: Too much boilerplate, inconsistent handling

**2. Configurable retry parameters** (e.g., `max_retries`, `backoff_multiplier`)
- **Pros**: More flexible, users can tune for their needs
- **Cons**: More complexity, more surface area for bugs
- **Rejected**: Current defaults work well for 95% of cases, can be added later if needed

**3. Retry at higher level** (e.g., in `aexecute` instead of `_aforward`/`_astream`)
- **Pros**: Simpler implementation, single retry point
- **Cons**: Would retry tool calls and other non-LLM logic unnecessarily
- **Rejected**: Parse errors only occur in LLM response parsing, not tool execution

**4. Use different retry library** (e.g., `backoff`, manual implementation)
- **Pros**: Potentially smaller dependency
- **Cons**: Tenacity is well-maintained, widely used, excellent async support
- **Rejected**: Tenacity is the industry standard for Python retry logic

### Testing Strategy

To keep tests fast, a global `fast_retry` fixture is used in `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def fast_retry():
    """Patch retry decorators to use no wait time for fast tests."""
    fast_retry_decorator = retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_none(),  # No wait between retries
    )

    with patch("udspy.module.predict.Predict._aforward",
               new=fast_retry_decorator(Predict._aforward.__wrapped__)):
        with patch("udspy.module.predict.Predict._astream",
                   new=fast_retry_decorator(Predict._astream.__wrapped__)):
            yield
```

This ensures:
- Tests run instantly (no exponential backoff wait times)
- Retry logic is still exercised in tests
- Production code uses proper backoff timings

### Migration Guide

**This is a non-breaking change** - no user code needs to be updated.

Users who previously implemented their own retry logic can remove it:

```python
# Before (manual retry)
for attempt in range(3):
    try:
        result = predictor(question="...")
        break
    except AdapterParseError:
        if attempt == 2:
            raise
        time.sleep(0.1 * (2 ** attempt))

# After (automatic retry)
result = predictor(question="...")  # Retry is automatic
```

### Future Considerations

1. **Make retry configurable**: Add `max_retries` parameter to `Predict.__init__()` if users need to tune it
2. **Add retry callback**: Allow users to hook into retry events for logging/metrics
3. **Smarter retry**: Analyze parse error type and adjust retry strategy (e.g., don't retry on schema validation errors that won't be fixed by retry)
4. **Retry budget**: Add global retry limit to prevent excessive token usage from many retries

---

## ADR-008: Module Callbacks and Dynamic Tool Management

**Date**: 2025-10-31

**Status**: Accepted

### Context

Agents often need specialized tools that should only be loaded on demand rather than being available from the start. Use cases include:
- Loading expensive or resource-intensive tools only when needed
- Progressive tool discovery (agent figures out what tools it needs as it works)
- Category-based tool loading (math tools, web tools, data tools)
- Multi-tenant applications with user-specific tool permissions
- Reducing initial token usage and context size

### Decision

Implement a module callback system where tools can return special callables decorated with `@module_callback` that modify the module's available tools during execution:

```python
from udspy import ReAct, tool, module_callback

@tool(name="calculator", description="Perform calculations")
def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

@tool(name="load_calculator", description="Load calculator tool")
def load_calculator() -> callable:
    """Load calculator tool dynamically."""

    @module_callback
    def add_calculator(context):
        # Get current tools (excluding built-ins)
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

### Implementation Details

1. **@module_callback Decorator**: Simple marker decorator that adds `__udspy_module_callback__` attribute
2. **Return Value Detection**: After tool execution, check `is_module_callback(result)`
3. **Context Objects**: Pass execution context to callbacks:
   - `ReactContext`: Includes trajectory history
   - `PredictContext`: Includes conversation history
   - `ModuleContext`: Base with module reference
4. **init_module() Pattern**: Unified method to reinitialize tools and regenerate signatures
5. **Tool Persistence**: Dynamically loaded tools remain available until module execution completes

### Key Features

1. **Decorator-based API**: Clean, explicit marking of module callbacks
2. **Full module access**: Callbacks can inspect and modify module state
3. **Works with all modules**: Predict, ChainOfThought, ReAct
4. **Observation return**: Callbacks return strings that appear in trajectory
5. **Type-safe**: Context objects provide proper type hints

### Use Cases

1. **On-demand capabilities**: Load expensive tools only when needed
   ```python
   agent = ReAct(Task, tools=[load_nlp_tools, load_vision_tools])
   ```

2. **Progressive discovery**: Agent discovers needed tools as it works
   ```python
   agent = ReAct(Task, tools=[load_tools])  # Figures out what's needed
   ```

3. **Multi-tenant**: Load user-specific tools based on permissions
   ```python
   @tool(name="load_user_tools")
   def load_user_tools(user_id: str) -> callable:
       @module_callback
       def add_tools(context):
           tools = get_tools_for_user(user_id)
           context.module.init_module(tools=tools)
           return f"Loaded tools for user {user_id}"
       return add_tools
   ```

4. **Category loading**: Load tool groups on demand
   ```python
   @tool(name="load_tools")
   def load_tools(category: str) -> callable:  # "math", "web", "data"
       @module_callback
       def add_category_tools(context):
           tools = get_tools_by_category(category)
           context.module.init_module(tools=current + tools)
           return f"Loaded {len(tools)} {category} tools"
       return add_category_tools
   ```

### Consequences

**Benefits**:
- Reduced token usage and context size (only load tools when needed)
- Adaptive agent behavior (discovers capabilities progressively)
- Clean API with decorator pattern
- Full module state access through context
- Works seamlessly with existing tool system
- Enables multi-tenant tool isolation

**Trade-offs**:
- Additional complexity in tool execution logic
- Must remember to return string from callbacks (for trajectory)
- Tool persistence requires new instance for fresh state
- Context objects add small memory overhead
- Learning curve for callback pattern

### Alternatives Considered

- **Direct module mutation**: Rejected due to lack of encapsulation and thread safety concerns
- **Event system**: Rejected as too complex and heavyweight for this use case
- **Plugin architecture**: Rejected as overkill for simple tool management
- **Configuration-based loading**: Rejected as less flexible than programmatic control

### Migration Guide

Feature is additive - existing code continues to work unchanged.

To use dynamic tools:

1. Define tools that return `@module_callback` decorated functions
2. Callbacks receive context and call `context.module.init_module(tools=[...])`
3. Return string observation from callback
4. Tool persists for remainder of module execution

**Example**:
```python
# Before: All tools loaded upfront
agent = ReAct(Task, tools=[calculator, search, weather, ...])

# After: Load tools on demand
agent = ReAct(Task, tools=[load_calculator, load_search, load_weather])
```

### See Also

- [Dynamic Tools Guide](../examples/dynamic_tools.md)
- [Module Callbacks API](../api/module_callback.md)
- [Tool Calling Guide](../examples/tool_calling.md)

---

## ADR-009: History Management with System Prompts

**Date**: 2025-10-31

**Status**: Accepted

### Context

Chat histories need special handling for system prompts to ensure they're always positioned first in the message list. Module behavior depends on having system instructions properly placed, and tools may manipulate histories during execution. Without dedicated management, it's easy to accidentally insert system prompts mid-conversation or lose them during history manipulation.

### Decision

Implement `History` class with dedicated `system_prompt` property that ensures system messages always appear first:

```python
from udspy import History

history = History()

# Add conversation messages
history.add_message(role="user", content="Hello")
history.add_message(role="assistant", content="Hi there!")

# System prompt always goes first, even if set later
history.system_prompt = "You are a helpful assistant"

messages = history.messages
# [{"role": "system", "content": "You are a helpful assistant"},
#  {"role": "user", "content": "Hello"},
#  {"role": "assistant", "content": "Hi there!"}]
```

### Implementation Details

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

**Key aspects**:
- System prompt stored separately from regular messages
- `messages` property dynamically constructs full list
- No risk of system prompt appearing mid-conversation
- Simple to update system prompt without rebuilding list
- Clear ownership (History manages system message)

### Key Features

1. **Dedicated system_prompt property**: Special handling for system messages
2. **Automatic positioning**: System prompt always first in messages list
3. **Mutable**: Can update system prompt at any time, position maintained
4. **Copy support**: `history.copy()` includes system prompt
5. **Clear separation**: Regular messages in `_messages`, system prompt separate

### Use Cases

1. **Module initialization**: Set system prompt per module type
   ```python
   history = History(system_prompt="You are a ReAct agent. Use tools to solve tasks.")
   ```

2. **Dynamic prompts**: Update based on context or user
   ```python
   history.system_prompt = f"You are assisting {user.name}. Use their preferences: {prefs}"
   ```

3. **Tool manipulation**: Tools can safely update system prompt
   ```python
   @tool(name="change_persona")
   def change_persona(persona: str) -> str:
       # Tool can access and modify history.system_prompt
       return f"Changed to {persona} persona"
   ```

4. **History replay**: Maintain system prompt across sessions
   ```python
   saved_history = history.to_dict()  # Save including system prompt
   loaded_history = History.from_dict(saved_history)  # Restore
   ```

5. **Multi-turn conversations**: System prompt persists correctly
   ```python
   # System prompt set once, remains first through all turns
   for user_msg in conversation:
       history.add_message(role="user", content=user_msg)
       # System prompt still first
   ```

### Consequences

**Benefits**:
- System prompt guaranteed to be first (LLM APIs require this)
- Can update system prompt at any time safely
- Clean property-based API
- Prevents common mistakes (system prompt mid-conversation)
- Supports all history manipulation patterns
- No manual list management required

**Trade-offs**:
- Small overhead constructing messages list on each access (negligible)
- System message can't be treated like regular message (by design)
- Slight complexity in History implementation vs. simple list
- Property access pattern may surprise developers expecting plain list

### Alternatives Considered

- **Insert at index 0**: Rejected as error-prone with mutations, easy to forget
- **Validation on add**: Rejected as too restrictive, doesn't prevent mid-conversation insertion
- **Separate system field in messages**: Rejected as doesn't integrate with standard message format
- **Manual management**: Status quo before this ADR, too error-prone

### Migration Guide

Existing code using `History.add_message()` continues to work unchanged.

To use system prompts:

**Create with system prompt**:
```python
history = History(system_prompt="You are a helpful assistant")
```

**Set later**:
```python
history = History()
# ... add messages ...
history.system_prompt = "You are a math tutor"
```

**Update dynamically**:
```python
history.system_prompt = f"You are assisting {user.name}"
```

**Always correctly positioned**:
```python
messages = history.messages  # System prompt is always first
```

### See Also

- [History API Reference](../api/history.md)
- [Module Architecture](modules.md)

---

## ADR-010: LM Callable Interface with String Prompts

**Date**: 2025-10-31

**Status**: Accepted

### Context

Users want the simplest possible interface for quick LLM queries without needing to construct message dictionaries. Common use cases include:
- Prototyping and experimentation
- Simple scripts and utilities
- Interactive sessions (REPL)
- Learning and onboarding new users
- Quick one-off queries

The existing API required constructing message lists even for simple prompts:
```python
response = lm.complete([{"role": "user", "content": "Hello"}], model="gpt-4o")
text = response.choices[0].message.content
```

### Decision

Enhanced LM base class to accept simple string prompts via `__call__()` and return just the text content:

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

### Implementation Details

```python
from typing import overload

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

**Key aspects**:
1. **Overloaded signatures**: `@overload` provides proper type hints for both modes
2. **Type-based dispatch**: `isinstance(prompt_or_messages, str)` determines behavior
3. **Message wrapping**: String prompts wrapped as `[{"role": "user", "content": prompt}]`
4. **Text extraction**: For strings, extract `response.choices[0].message.content`
5. **Fallback**: If extraction fails, fall back to `str(response)`
6. **Optional model**: Made `model` parameter optional everywhere, uses `self.model` as default

### Key Features

1. **Two modes**:
   - String input → returns text only (str)
   - Messages list → returns full response object (Any)
2. **Type-safe**: Proper overloads for IDE autocomplete
3. **Backward compatible**: Existing message-list usage unchanged
4. **Optional model**: Falls back to instance's default model
5. **Passes kwargs**: Temperature, max_tokens, etc. work in both modes

### Use Cases

1. **Prototyping**: Quick tests without boilerplate
   ```python
   answer = lm("Is Python interpreted or compiled?")
   ```

2. **Simple scripts**: One-line LLM queries
   ```python
   summary = lm(f"Summarize this in one sentence: {long_text}")
   ```

3. **Interactive sessions**: REPL-friendly API
   ```python
   >>> lm("What's 2+2?")
   '4'
   ```

4. **Learning**: Easiest API for newcomers
   ```python
   # First udspy program
   lm = OpenAILM(client, "gpt-4o-mini")
   print(lm("Hello!"))
   ```

5. **Utilities**: Helper functions
   ```python
   def translate(text: str, target_lang: str) -> str:
       return lm(f"Translate to {target_lang}: {text}")
   ```

### Consequences

**Benefits**:
- Simplest possible API for common case (string prompt)
- No need to construct message dictionaries
- Backward compatible with existing code
- Proper type hints for IDE support (overloads)
- Falls back gracefully if text extraction fails
- Model parameter now optional everywhere

**Trade-offs**:
- Slight complexity in `__call__` implementation (type dispatch)
- String/list dispatch adds minor overhead (negligible)
- Text extraction logic specific to OpenAI response format
- Two different return types require overloads for type safety
- Can't use tools or streaming with string prompt mode

### Alternatives Considered

- **Separate method** (`lm.ask("prompt")`): Rejected as less convenient, extra method to learn
- **Always return text**: Rejected as losing access to full response metadata
- **Factory function**: Rejected as less object-oriented, doesn't fit with LM abstraction
- **Auto-detect return type**: Rejected as confusing, breaks type safety

### Migration Guide

No migration needed - feature is additive and backward compatible.

**Before (verbose)**:
```python
response = lm.complete([{"role": "user", "content": "Hello"}], model="gpt-4o")
text = response.choices[0].message.content
```

**After (concise)**:
```python
text = lm("Hello", model="gpt-4o")
```

**Still supported (full control)**:
```python
response = lm(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",
    tools=[...],
    stream=True
)
```

### See Also

- [Basic Usage Guide](../examples/basic_usage.md)
- [LM Architecture](lm.md)

---

## Template for Future ADRs

When adding new architectural decisions, use this template:

## ADR-XXX: Decision Title

**Date**: YYYY-MM-DD

**Status**: Proposed | Accepted | Deprecated | Superseded

### Context

Why was this change needed? What problem does it solve?

### Decision

What was decided and implemented? Include code examples if relevant.

### Implementation Details

How is this implemented? Key technical details.

### Consequences

**Benefits**:
- What are the advantages?

**Trade-offs**:
- What are the disadvantages or limitations?

### Alternatives Considered

- What other approaches were considered?
- Why were they rejected?

### Migration Guide (if applicable)

How should users update their code?
