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

---

## ADR-001: Initial Project Setup

**Date**: 2025-10-24

**Status**: Accepted

### Context

Created a minimal DSPy-inspired library focused on resource-constrained environments where DSPy's ~200MB memory footprint (due to LiteLLM) is prohibitive.

### Decision

Build a lightweight alternative with:
- Native OpenAI tool calling instead of custom adapters
- Minimal dependencies (~10MB: `openai` + `pydantic`)
- Streaming support for reasoning and output fields
- Modern Python tooling (uv, ruff, justfile)

### Key Design Decisions

#### 1. Native Tool Calling

Use OpenAI's native function calling API instead of custom adapters and field markers.

**Rationale**:
- OpenAI's tool calling is optimized and well-tested
- Reduces complexity and leverages provider's optimizations
- Better reliability for structured tool invocation
- Forward compatible with future improvements

**Trade-offs**:
- Couples to OpenAI's API format (but works with any OpenAI-compatible provider)
- May need adapters for other providers in future

#### 2. Minimal Dependencies

Only `openai` and `pydantic` in core dependencies.

**Rationale**:
- Keeps the library lightweight and maintainable (~10MB vs ~200MB)
- Reduces potential dependency conflicts
- Faster installation and lower memory usage
- Suitable for serverless, edge, and embedded deployments

**Trade-offs**:
- Can't leverage broader ecosystem for advanced features
- Users need to install extras for dev tools

#### 3. Pydantic v2

Use Pydantic v2 for all models and validation.

**Rationale**:
- Modern, fast, well-maintained
- Excellent JSON schema generation for tools
- Built-in validation and type coercion
- Better performance than v1
- Great developer experience with IDE support

**Trade-offs**:
- Requires Python 3.7+ (we target 3.11+)

#### 4. Streaming Architecture

Async-first design using Python's async/await.

**Rationale**:
- Python's async is the standard for I/O-bound operations
- Native support from OpenAI SDK
- Better composability with other async code
- Easier to reason about than callbacks

**Trade-offs**:
- Requires async runtime (asyncio)
- Steeper learning curve for beginners

#### 5. Module Abstraction

Modules compose via Python class inheritance.

**Rationale**:
- Similar to DSPy but simplified
- Familiar Python patterns (no custom DSL)
- Good IDE and type checker support
- Signatures define I/O contracts using Pydantic models
- Predict is the core primitive for LLM calls

**Trade-offs**:
- Less "magical" than DSPy's meta-programming
- Requires more explicit code

### Consequences

**Benefits**:
- 20x smaller memory footprint (~10MB vs ~200MB)
- Works in resource-constrained environments
- Simple, maintainable codebase
- Compatible with any OpenAI-compatible provider

**Trade-offs**:
- Less feature-complete than DSPy
- Fewer LLM providers supported out-of-the-box
- No built-in optimizers or teleprompters

### Alternatives Considered

- **Fork DSPy**: Too much baggage and complexity
- **Use LangChain**: Even larger footprint, different philosophy
- **Build from scratch**: Chose this - learn from DSPy's excellent patterns

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

### Comparison with DSPy

| Aspect | udspy | DSPy |
|--------|-------|------|
| API | `ChainOfThought(signature)` | `dspy.ChainOfThought(signature, rationale_field=...)` |
| Implementation | Dynamic signature creation | Signature.prepend() method |
| Customization | `reasoning_description` param | Full `rationale_field` control |
| Complexity | ~45 lines | ~40 lines |
| Dependencies | Uses `make_signature` | Uses signature mutation |

Both are equally effective; udspy's approach is simpler but less flexible in edge cases.

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

**Date**: 2025-10-25

**Status**: Accepted

### Context

Many agent applications require human approval for certain actions (e.g., deleting files, sending emails, making purchases). We needed a clean way to suspend execution, ask for user input, and resume where we left off.

### Decision

Implement an `@confirm_first` decorator that:
- Suspends function execution before it runs
- Raises `ConfirmationRequired` exception with context
- Allows resumption with user approval/rejection/modifications
- Uses thread-safe `contextvars` for state management
- Generates stable confirmation IDs based on function + arguments

```python
from udspy import confirm_first, ConfirmationRequired, respond_to_confirmation

@confirm_first
def delete_file(path: str) -> str:
    os.remove(path)
    return f"Deleted {path}"

try:
    delete_file("/important.txt")
except ConfirmationRequired as e:
    # User approves
    respond_to_confirmation(e.confirmation_id, approved=True)
    delete_file("/important.txt")  # Now executes
```

### Implementation Details

1. **Stable Confirmation IDs**: Generated from `function_name:hash(args)` to allow same call to resume
2. **Context Variables**: Thread-safe and async-safe state storage
3. **Rejection Support**: `ConfirmationRejected` exception distinguishes "user said no" from "pending"
4. **Argument Modification**: Users can edit arguments before approval
5. **Automatic Cleanup: Confirmations are cleared after successful execution

### Key Features

1. **Decorator-based**: Simple to apply to any function
2. **Thread-safe**: Works with concurrent requests
3. **Async-safe**: Works with asyncio tasks
4. **Resumable**: Same function call can be resumed after approval
5. **Integrated with Tools**: Works seamlessly with `@tool` decorator

### Use Cases

1. **Dangerous Operations**: File deletion, system commands
2. **User Confirmation**: Sending emails, making purchases
3. **Clarification**: Ask user for additional information
4. **Argument Editing**: Let user modify parameters before execution

### Consequences

**Benefits**:
- Clean separation of business logic from approval logic
- Works naturally with ReAct agent workflows
- Thread-safe and async-safe out of the box
- Easy to test (deterministic based on confirmation state)

**Trade-offs**:
- Requires exception handling (but this is explicit and clear)
- Confirmation state needs to be managed (cleared on success)
- Not suitable for purely synchronous, single-threaded apps (but works fine there too)

### Alternatives Considered

- **Callback-based**: More complex, harder to reason about flow
- **Middleware pattern**: Too heavyweight for this use case
- **Manual state management**: Error-prone, not thread-safe

### Migration Guide

Feature is additive - no migration needed.

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

### Comparison with DSPy

| Aspect | udspy | DSPy |
|--------|-------|------|
| API | `ReAct(signature, tools=[...])` | `dspy.ReAct(signature, tools=[...])` |
| Human-in-Loop | Built-in with `@confirm_first` | External handling |
| Streaming | Supported | Limited |
| Tool Execution | Automatic with error handling | Automatic |
| Max Iterations | Configurable with `max_iters` | Configurable |

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
