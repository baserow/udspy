"""Tests for chat adapter."""

from enum import Enum
from typing import Literal

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from udspy import ChatAdapter, InputField, OutputField, Signature
from udspy.adapter import parse_value, translate_field_type
from udspy.exceptions import AdapterParseError
from udspy.streaming import OutputStreamChunk
from udspy.tool.types import ToolCall


def test_format_instructions() -> None:
    """Test formatting signature instructions."""

    class QA(Signature):
        """Answer questions concisely."""

        question: str = InputField(description="Question to answer")
        answer: str = OutputField(description="Concise answer")

    adapter = ChatAdapter()
    instructions = adapter.format_instructions(QA)

    assert "Answer questions concisely" in instructions
    assert "question" in instructions
    assert "answer" in instructions
    # Field descriptions are no longer in system message (moved to output instructions)
    assert "Given the input fields" in instructions
    assert "produce the output fields" in instructions


def test_format_inputs() -> None:
    """Test formatting input values."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    inputs = {"question": "What is 2+2?"}
    formatted = adapter.format_inputs(QA, inputs)

    assert "[[ ## question ## ]]" in formatted
    assert "What is 2+2?" in formatted


def test_parse_outputs() -> None:
    """Test parsing JSON LLM outputs."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    completion = '{"answer": "4"}'

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "4"


def test_parse_outputs_with_multiple_fields() -> None:
    """Test parsing multiple output fields from JSON."""

    class Reasoning(Signature):
        """Task with reasoning."""

        query: str = InputField()
        reasoning: str = OutputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    completion = '{"reasoning": "Let me think about this...", "answer": "The answer is 42"}'

    outputs = adapter.parse_outputs(Reasoning, completion)

    assert "reasoning" in outputs
    assert "answer" in outputs
    assert "Let me think" in outputs["reasoning"]
    assert "42" in outputs["answer"]


def test_format_tool_schemas() -> None:
    """Test converting Pydantic models to OpenAI tool schemas."""

    class Calculator(BaseModel):
        """Perform arithmetic operations."""

        operation: str
        a: float
        b: float

    adapter = ChatAdapter()
    schemas = adapter.format_tool_schemas([Calculator])

    assert len(schemas) == 1
    schema = schemas[0]

    assert schema["type"] == "function"
    assert "Calculator" in schema["function"]["name"]
    assert "arithmetic" in schema["function"]["description"].lower()
    assert "operation" in schema["function"]["parameters"]["properties"]
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["properties"]


def test_parse_value_with_different_types() -> None:
    """Test parse_value handles different types correctly."""

    # Test int
    assert parse_value("42", int) == 42

    # Test float
    assert parse_value("3.14", float) == 3.14

    # Test bool
    assert parse_value("true", bool) is True
    assert parse_value("yes", bool) is True
    assert parse_value("1", bool) is True
    assert parse_value("false", bool) is False

    # Test list
    assert parse_value("[1, 2, 3]", list) == [1, 2, 3]

    # Test dict
    assert parse_value('{"key": "value"}', dict) == {"key": "value"}

    # Test str (default)
    assert parse_value("hello", str) == "hello"

    # Test fallback to string for unknown type
    assert parse_value("anything", object) == "anything"


def test_parse_value_with_pydantic_model() -> None:
    """Test parse_value handles Pydantic models."""

    class TestModel(BaseModel):
        name: str
        age: int

    # Test with JSON object
    result = parse_value('{"name": "Alice", "age": 30}', TestModel)
    assert isinstance(result, TestModel)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_outputs_with_extra_json_whitespace() -> None:
    """Test that parse_outputs handles JSON with extra whitespace."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # JSON with extra whitespace
    completion = '{\n  "answer": "Paris"\n}'

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "Paris"


def test_parse_outputs_invalid_json_raises_error() -> None:
    """Test that parse_outputs raises AdapterParseError for invalid JSON."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # Invalid JSON
    completion = "Not JSON at all"

    with pytest.raises(AdapterParseError):
        adapter.parse_outputs(QA, completion)


def test_parse_outputs_preserves_multiline_strings() -> None:
    """Test that parse_outputs preserves newlines in JSON string values."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # Multi-line answer in JSON
    completion = '{"answer": "Line 1\\nLine 2\\nLine 3"}'

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "Line 1\nLine 2\nLine 3"


def test_translate_field_type_string() -> None:
    """Test translate_field_type with string fields."""

    field_info = FieldInfo(annotation=str)
    result = translate_field_type("answer", field_info)

    # Strings should not have type constraints
    assert result == "{answer}"
    assert "note:" not in result


def test_translate_field_type_int() -> None:
    """Test translate_field_type with int fields."""

    field_info = FieldInfo(annotation=int)
    result = translate_field_type("count", field_info)

    assert "{count}" in result
    assert "must be a single int value" in result


def test_translate_field_type_float() -> None:
    """Test translate_field_type with float fields."""

    field_info = FieldInfo(annotation=float)
    result = translate_field_type("score", field_info)

    assert "{score}" in result
    assert "must be a single float value" in result


def test_translate_field_type_bool() -> None:
    """Test translate_field_type with bool fields."""

    field_info = FieldInfo(annotation=bool)
    result = translate_field_type("is_valid", field_info)

    assert "{is_valid}" in result
    assert "must be True or False" in result


def test_translate_field_type_enum() -> None:
    """Test translate_field_type with Enum fields."""

    class Priority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    field_info = FieldInfo(annotation=Priority)
    result = translate_field_type("priority", field_info)

    assert "{priority}" in result
    assert "must be one of:" in result
    assert "low" in result
    assert "medium" in result
    assert "high" in result


def test_translate_field_type_literal() -> None:
    """Test translate_field_type with Literal fields."""

    field_info = FieldInfo(annotation=Literal["pending", "approved", "rejected"])
    result = translate_field_type("status", field_info)

    assert "{status}" in result
    assert "must exactly match (no extra characters) one of:" in result
    assert "pending" in result
    assert "approved" in result
    assert "rejected" in result


def test_translate_field_type_pydantic_model() -> None:
    """Test translate_field_type with Pydantic model fields."""

    class Person(BaseModel):
        name: str
        age: int

    field_info = FieldInfo(annotation=Person)
    result = translate_field_type("person", field_info)

    assert "{person}" in result
    assert "must adhere to the JSON schema:" in result
    assert "properties" in result
    assert "name" in result
    assert "age" in result


def test_format_field_structure_basic() -> None:
    """Test format_field_structure with basic signature."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    structure = adapter.format_field_structure(QA)

    assert "All interactions will be structured" in structure
    assert "[[ ## question ## ]]" in structure
    assert "[[ ## answer ## ]]" in structure
    assert "[[ ## completed ## ]]" in structure
    assert "{question}" in structure
    assert "{answer}" in structure


def test_format_field_structure_with_types() -> None:
    """Test format_field_structure with various field types."""

    class MathQA(Signature):
        """Math question answering."""

        question: str = InputField()
        reasoning: str = OutputField()
        answer: int = OutputField()

    adapter = ChatAdapter()
    structure = adapter.format_field_structure(MathQA)

    # Check for type hints
    assert "{question}" in structure
    assert "{reasoning}" in structure
    assert "{answer}" in structure
    assert "must be a single int value" in structure


def test_format_field_structure_multiple_complex_types() -> None:
    """Test format_field_structure with multiple complex types."""

    class Priority(Enum):
        LOW = "low"
        HIGH = "high"

    class TaskAnalysis(Signature):
        """Analyze tasks."""

        task: str = InputField()
        is_urgent: bool = OutputField()
        priority: Priority = OutputField()
        estimated_hours: float = OutputField()

    adapter = ChatAdapter()
    structure = adapter.format_field_structure(TaskAnalysis)

    # Check all type hints are present
    assert "must be True or False" in structure
    assert "must be one of: low; high" in structure
    assert "must be a single float value" in structure


def test_format_instructions_includes_field_structure() -> None:
    """Test that format_instructions includes only task description and field names."""

    class MathQA(Signature):
        """Answer math questions."""

        question: str = InputField(description="Math question")
        answer: int = OutputField(description="Numeric answer")

    adapter = ChatAdapter()
    instructions = adapter.format_instructions(MathQA)

    # Should include the signature description
    assert "Answer math questions" in instructions

    # Should include field names in the "Given..." sentence
    assert "question" in instructions
    assert "answer" in instructions
    assert "Given the input fields" in instructions
    assert "produce the output fields" in instructions

    # Should NOT include field structure (that's now in format_output_instructions)
    assert "[[ ## question ## ]]" not in instructions
    assert "[[ ## answer ## ]]" not in instructions


def test_format_output_instructions() -> None:
    """Test that format_output_instructions generates proper output instructions."""

    class MathQA(Signature):
        """Answer math questions."""

        question: str = InputField(description="Math question")
        answer: int = OutputField(description="Numeric answer")

    adapter = ChatAdapter()
    output_instructions = adapter.format_output_instructions(MathQA)

    # Should include instructions to respond with JSON
    assert "JSON object" in output_instructions
    assert "`answer`" in output_instructions
    assert "must be a single int value" in output_instructions


def test_format_user_request() -> None:
    """Test that format_user_request combines inputs and output instructions."""

    class MathQA(Signature):
        """Answer math questions."""

        question: str = InputField(description="Math question")
        answer: int = OutputField(description="Numeric answer")

    adapter = ChatAdapter()
    user_request = adapter.format_user_request(MathQA, {"question": "What is 2+2?"})

    # Should include formatted input
    assert "[[ ## question ## ]]" in user_request
    assert "What is 2+2?" in user_request

    # Should include JSON output instructions
    assert "JSON object" in user_request
    assert "`answer`" in user_request
    assert "must be a single int value" in user_request


@pytest.mark.asyncio
async def test_process_chunk_and_finalize() -> None:
    """Test process_chunk and finalize methods for streaming responses."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Create mock module
    class MockModule:
        pass

    module = MockModule()
    adapter = ChatAdapter()

    # Create streaming chunks
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content='{"answer": "', role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="test",
            model="gpt-4o",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(index=0, delta=ChoiceDelta(content="Paris", role=None), finish_reason=None)
            ],
        ),
        ChatCompletionChunk(
            id="test",
            model="gpt-4o",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(index=0, delta=ChoiceDelta(content='"}', role=None), finish_reason="stop")
            ],
        ),
    ]

    # Process chunks
    events = []
    for chunk in chunks:
        async for event in adapter.process_chunk(chunk, module, QA):
            events.append(event)

    # Should have yielded OutputStreamChunk events
    assert len(events) > 0
    assert any(isinstance(e, OutputStreamChunk) for e in events)

    # Finalize and get outputs
    outputs, native_tool_calls, _completion_text = await adapter.finalize(QA)

    assert outputs["answer"] == "Paris"
    assert native_tool_calls == []


@pytest.mark.asyncio
async def test_reset_parser_between_requests() -> None:
    """Test that reset_parser clears state between requests."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    class MockModule:
        pass

    module = MockModule()
    adapter = ChatAdapter()

    # First request
    chunk1 = ChatCompletionChunk(
        id="test1",
        model="gpt-4o",
        object="chat.completion.chunk",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content='{"answer": "First"}', role="assistant"),
                finish_reason="stop",
            )
        ],
    )

    async for _event in adapter.process_chunk(chunk1, module, QA):
        pass

    outputs1, _, _ = await adapter.finalize(QA)
    assert outputs1["answer"] == "First"

    # Parser should be reset after finalize
    assert adapter._streaming_parser is None

    # Second request should work independently
    chunk2 = ChatCompletionChunk(
        id="test2",
        model="gpt-4o",
        object="chat.completion.chunk",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content='{"answer": "Second"}', role="assistant"),
                finish_reason="stop",
            )
        ],
    )

    async for _event in adapter.process_chunk(chunk2, module, QA):
        pass

    outputs2, _, _ = await adapter.finalize(QA)
    assert outputs2["answer"] == "Second"


@pytest.mark.asyncio
async def test_validate_outputs_raises_on_mismatch() -> None:
    """Test that validate_outputs raises AdapterParseError when outputs don't match signature."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()

    # Missing required field
    with pytest.raises(AdapterParseError) as exc_info:
        adapter.validate_outputs(QA, {}, [], '{"wrong": "field"}')

    assert "answer" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_outputs_allows_tool_calls_without_outputs() -> None:
    """Test that validate_outputs allows tool calls without output fields."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()

    # Tool calls present, so empty outputs is OK
    tool_calls = [ToolCall(call_id="call_1", name="search", args={"query": "test"})]

    # Should not raise
    adapter.validate_outputs(QA, {}, tool_calls, "")
