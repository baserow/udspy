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
