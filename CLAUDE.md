# Architectural Changes

This document tracks major architectural decisions and changes made to udspy.

## 2025-01-24: Initial Project Setup

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
