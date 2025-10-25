"""Tests for chat adapter."""

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
