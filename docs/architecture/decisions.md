# Architectural Decision Records (ADR)

This document tracks major architectural decisions made in udspy, presented chronologically with context, rationale, and consequences.

## Table of Contents

1. [Initial Project Setup (2025-01-24)](#adr-001-initial-project-setup)
2. [Context Manager for Settings (2025-01-24)](#adr-002-context-manager-for-settings)
3. [Chain of Thought Module (2025-01-24)](#adr-003-chain-of-thought-module)

---

## ADR-001: Initial Project Setup

**Date**: 2025-01-24

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

**Date**: 2025-01-24

**Status**: Accepted

### Context

Need to support different API keys and models in different contexts (e.g., multi-tenant apps, different users, testing scenarios, concurrent async operations).

### Decision

Implement thread-safe context manager using Python's `contextvars` module:

```python
# Global settings
udspy.settings.configure(api_key="global-key", model="gpt-4o-mini")

# Temporary override in context
with udspy.settings.context(api_key="user-key", model="gpt-4"):
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
3. **Comprehensive**: Supports overriding api_key, model, client, async_client, and any kwargs
4. **Clean API**: Simple context manager interface
5. **Backwards Compatible**: Existing code continues to work without changes

### Use Cases

1. **Multi-tenant applications**: Different API keys per user
   ```python
   with udspy.settings.context(api_key=user.api_key):
       result = predictor(question=user.question)
   ```

2. **Model selection per request**: Use different models for different tasks
   ```python
   with udspy.settings.context(model="gpt-4"):
       result = expensive_predictor(question=complex_question)
   ```

3. **Testing**: Isolate test settings without affecting global state
   ```python
   with udspy.settings.context(api_key="sk-test", temperature=0.0):
       assert predictor(question="2+2").answer == "4"
   ```

4. **Async operations**: Safe concurrent operations with different settings
   ```python
   async def handle_user(user):
       with udspy.settings.context(api_key=user.api_key):
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

**Date**: 2025-01-24

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
