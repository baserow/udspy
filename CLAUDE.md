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
