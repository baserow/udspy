# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8] - 2025-10-31

### Changed
- Lazy import of `openai` to reduce memory footprint
- Simplified callback inputs and added telemetry support

## [0.1.7] - 2025-10-31

### Added
- ReAct now passes kwargs through to underlying Predict modules

### Changed
- Wrapped OpenAI API calls in `_acomplete` for tracing support

## [0.1.6] - 2025-10-31

### Added
- Comprehensive provider configuration guide for LM
- Improved environment variable precedence for multi-provider support

### Changed
- Simplified LM factory implementation
- Updated env var name to `UDSPY_LM_OPENAI_COMPATIBLE_BASE_URL`
- Removed unused history parameter from ReactContext

### Fixed
- API key precedence documentation to match implementation

## [0.1.5] - 2025-10-31

### Added
- Auto-initialize LM from `UDSPY_LM_MODEL` environment variable
- History tracking integration in ReAct module
- Comprehensive version validation for releases

## [0.1.4] - 2025-10-31

### Changed
- Use `importlib.metadata` for version instead of hardcoded string

### Fixed
- Type annotations to resolve mypy errors
- Adapter `remaining_content` initialization before conditional assignment

## [0.1.3] - 2025-10-31

### Added
- LM abstraction layer with registry-based factory and multi-provider support (OpenAI, Groq, Bedrock, Ollama)
- LM callable interface: `lm("prompt")` returns text directly, `lm(messages)` returns full response
- DSPy-compatible callback system for telemetry and monitoring
- Confirmation system (renamed from interrupt system) with `ConfirmationRequired`, `ConfirmationRejected`, `ResumeState`
- `set_system_message()` for guaranteed system prompt positioning in History
- `module` and `is_final` attributes on Prediction for nested module support
- String signature support: `Predict("question -> answer")`
- jiter-based JSON streaming parser for adapter
- `parse_and_validate_args` for Pydantic model arguments in tools
- `run_async_with_context` for context preservation in async operations
- Dynamic tool management documentation and examples

### Changed
- **Breaking**: Settings API simplified to only accept LM instances (`configure(lm=...)` instead of `configure(api_key=..., model=...)`)
- **Breaking**: Restructured ReAct trajectory as `list[Episode]` with typed fields (`thought`, `tool_name`, `tool_args`, `observation`)
- Renamed `_aexecute` to `aexecute` for public API
- Split `tool.py` into `tool/` package
- Split `predict.py` into focused submodules
- Split `utils` into package with `async_support.py`, `formatting.py`, `schema.py`
- Removed `GroqLM` class in favor of OpenAI-compatible API
- Made `emit_event` synchronous with `put_nowait`

### Fixed
- Streaming queue consumption prioritization
- Tool callback timing moved to actual execution point
- React tool execution `UnboundLocalError`
- Handle models without provider prefix in `_clean_model_name`

## [0.1.2] - 2025-10-24

### Added
- Codecov integration with OIDC tokenless upload
- Pre-release check command

### Fixed
- Documentation badge URLs and absolute paths

## [0.1.1] - 2025-10-24

### Added
- ReAct module with native tool calling and human-in-the-loop flow
- ChainOfThought module for step-by-step reasoning
- History class for conversation management
- Context manager for thread-safe settings overrides
- Automatic tool calling with streaming support
- Optional tool execution with `auto_execute_tools` parameter
- Architectural Decision Records (ADRs)
- GitHub Actions CI/CD workflows

### Changed
- Converted `module.py` into `module/` package
- Standardized module execution with `_aexecute` pattern
- Improved field parsing with regex-based extraction

## [0.1.0] - 2025-10-24

### Added
- Core Predict module with streaming support
- Signature system for defining inputs/outputs
- ChatAdapter for OpenAI integration
- Tool calling with `@tool` decorator
- Async-first architecture with sync wrappers
- Streaming with field-level chunking
- Event system for custom streaming events
- Native OpenAI tool calling integration
- Pydantic-based schemas
- Type-safe API with full type hints
- Comprehensive test suite
- Documentation with MkDocs
