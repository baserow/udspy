# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- History class for conversation management
- Optional tool execution with `auto_execute_tools` parameter
- Comprehensive documentation and examples

### Changed
- Renamed example files to remove redundant "_example" suffix

### Fixed
- Type checking error in settings.py for default_model property

## [0.1.0] - Initial Release

### Added
- Core Predict module with streaming support
- Signature system for defining inputs/outputs
- ChatAdapter for OpenAI integration
- ChainOfThought module for step-by-step reasoning
- Tool calling with `@tool` decorator
- Automatic multi-turn conversation handling
- Context-aware settings management
- Comprehensive test suite (90%+ coverage)
- Documentation with MkDocs

### Features
- Async-first architecture with sync wrappers
- Streaming with field-level chunking
- Event system for custom streaming events
- Native OpenAI tool calling integration
- Pydantic-based schemas
- Type-safe API with full type hints
