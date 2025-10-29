"""Tests for chat adapter."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from udspy import ChatAdapter, InputField, OutputField, Signature


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
    assert "Question to answer" in instructions
    assert "Concise answer" in instructions


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
    """Test parsing LLM outputs."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    completion = "[[ ## answer ## ]]\n4"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "4"


def test_parse_outputs_with_multiple_fields() -> None:
    """Test parsing multiple output fields."""

    class Reasoning(Signature):
        """Task with reasoning."""

        query: str = InputField()
        reasoning: str = OutputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    completion = (
        "[[ ## reasoning ## ]]\nLet me think about this...\n[[ ## answer ## ]]\nThe answer is 42"
    )

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
    from udspy.adapter import parse_value

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
    from udspy.adapter import parse_value

    class TestModel(BaseModel):
        name: str
        age: int

    # Test with JSON object
    result = parse_value('{"name": "Alice", "age": 30}', TestModel)
    assert isinstance(result, TestModel)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_outputs_with_extra_text_before_marker() -> None:
    """Test that parse_outputs ignores text before field markers."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # LLM adds preamble before the marker
    completion = "Let me answer your question.\n\n[[ ## answer ## ]]\nParis"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "Paris"


def test_parse_outputs_with_extra_text_after_marker() -> None:
    """Test that parse_outputs ignores text after field content."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # LLM adds extra text after the answer
    completion = "[[ ## answer ## ]]\nParis\n\nI hope this helps!"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    # Should only capture "Paris", not the extra text
    assert outputs["answer"] == "Paris\n\nI hope this helps!"


def test_parse_outputs_strips_newlines() -> None:
    """Test that parse_outputs strips leading/trailing newlines from content."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # Extra newlines after marker and before content
    completion = "[[ ## answer ## ]]\n\n\nParis\n\n\n"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    # Should strip leading/trailing newlines
    assert outputs["answer"] == "Paris"


def test_parse_outputs_preserves_internal_newlines() -> None:
    """Test that parse_outputs preserves newlines within content."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # Multi-line answer with internal newlines
    completion = "[[ ## answer ## ]]\nLine 1\nLine 2\nLine 3"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "Line 1\nLine 2\nLine 3"


def test_parse_outputs_with_varied_whitespace_in_markers() -> None:
    """Test that parse_outputs handles varied whitespace in markers."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    adapter = ChatAdapter()
    # Marker with extra spaces
    completion = "[[  ##  answer  ##  ]]\nParis"

    outputs = adapter.parse_outputs(QA, completion)

    assert "answer" in outputs
    assert outputs["answer"] == "Paris"


def test_translate_field_type_string() -> None:
    """Test translate_field_type with string fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

    field_info = FieldInfo(annotation=str)
    result = translate_field_type("answer", field_info)

    # Strings should not have type constraints
    assert result == "{answer}"
    assert "note:" not in result


def test_translate_field_type_int() -> None:
    """Test translate_field_type with int fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

    field_info = FieldInfo(annotation=int)
    result = translate_field_type("count", field_info)

    assert "{count}" in result
    assert "must be a single int value" in result


def test_translate_field_type_float() -> None:
    """Test translate_field_type with float fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

    field_info = FieldInfo(annotation=float)
    result = translate_field_type("score", field_info)

    assert "{score}" in result
    assert "must be a single float value" in result


def test_translate_field_type_bool() -> None:
    """Test translate_field_type with bool fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

    field_info = FieldInfo(annotation=bool)
    result = translate_field_type("is_valid", field_info)

    assert "{is_valid}" in result
    assert "must be True or False" in result


def test_translate_field_type_enum() -> None:
    """Test translate_field_type with Enum fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

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
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

    field_info = FieldInfo(annotation=Literal["pending", "approved", "rejected"])
    result = translate_field_type("status", field_info)

    assert "{status}" in result
    assert "must exactly match (no extra characters) one of:" in result
    assert "pending" in result
    assert "approved" in result
    assert "rejected" in result


def test_translate_field_type_pydantic_model() -> None:
    """Test translate_field_type with Pydantic model fields."""
    from pydantic.fields import FieldInfo

    from udspy.adapter import translate_field_type

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
    """Test that format_instructions includes field structure with type hints."""

    class MathQA(Signature):
        """Answer math questions."""

        question: str = InputField(description="Math question")
        answer: int = OutputField(description="Numeric answer")

    adapter = ChatAdapter()
    instructions = adapter.format_instructions(MathQA)

    # Should include the signature description
    assert "Answer math questions" in instructions

    # Should include field descriptions
    assert "Math question" in instructions
    assert "Numeric answer" in instructions

    # Should include field structure with type hints
    assert "[[ ## question ## ]]" in instructions
    assert "[[ ## answer ## ]]" in instructions
    assert "must be a single int value" in instructions
    assert "[[ ## completed ## ]]" in instructions
