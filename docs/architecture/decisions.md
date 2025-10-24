# Design Decisions

This document explains key architectural decisions made in udspy.

## Native Tool Calling

**Decision**: Use OpenAI's native function calling instead of custom prompt-based tools.

**Rationale**:
- OpenAI's tool calling is optimized and well-tested
- Reduces prompt complexity and token usage
- Better reliability for structured tool invocation
- Forward compatible with future OpenAI improvements

**Trade-offs**:
- Couples to OpenAI's API (less provider-agnostic)
- May need adapters for other providers in future

**Alternative Considered**: Custom field markers like DSPy's adapter system

---

## Pydantic v2 for Models

**Decision**: Use Pydantic v2 exclusively for all data modeling.

**Rationale**:
- Modern, fast, well-maintained
- Excellent JSON schema generation for tools
- Built-in validation and type coercion
- Great developer experience with IDE support

**Trade-offs**:
- Breaking changes from Pydantic v1 (not an issue for new project)
- Requires Python 3.7+ (we target 3.11+)

---

## Async-First Streaming

**Decision**: Streaming uses async/await, not callbacks or threads.

**Rationale**:
- Python's async is the standard for I/O-bound operations
- Better composability with other async code
- Easier to reason about than callbacks
- Native support from OpenAI SDK

**Trade-offs**:
- Requires async runtime (asyncio)
- Steeper learning curve for beginners

**Alternative Considered**: Synchronous generator with threads

---

## Field Markers for Parsing

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

**Alternative Considered**: JSON-only output format (less readable in prompts)

---

## Minimal Dependencies

**Decision**: Core library only depends on `openai` and `pydantic`.

**Rationale**:
- Easier to maintain and debug
- Faster installation
- Fewer security vulnerabilities
- Clearer responsibility boundaries

**Trade-offs**:
- Users need to install extras for dev tools
- Can't leverage ecosystem for advanced features

---

## Module Composition

**Decision**: Modules compose via Python class inheritance and composition.

**Rationale**:
- Familiar Python patterns
- No custom DSL to learn
- Good IDE and type checker support
- Easy to test and mock

**Trade-offs**:
- Less "magical" than DSPy's meta-programming
- Requires more explicit code

---

## Settings as Global Singleton

**Decision**: Configuration via global `settings` object.

**Rationale**:
- Convenient for most use cases
- Matches Django/Flask patterns
- Easy to override per-module
- Thread-safe with context managers (future)

**Trade-offs**:
- Global state can complicate testing
- Not ideal for multi-tenant applications

**Alternative Considered**: Dependency injection (more verbose)

---

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Chronological architectural changes
- [Architecture Overview](overview.md) - Component relationships
