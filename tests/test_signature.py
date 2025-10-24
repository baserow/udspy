"""Tests for signature definitions."""

import pytest

from udspy import InputField, OutputField, Signature, make_signature


def test_signature_with_input_and_output_fields() -> None:
    """Test basic signature with input and output fields."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField(description="Question to answer")
        answer: str = OutputField(description="Concise answer")

    assert QA.get_instructions() == "Answer questions."
    assert "question" in QA.get_input_fields()
    assert "answer" in QA.get_output_fields()


def test_signature_field_validation() -> None:
    """Test that fields must be declared with InputField or OutputField."""

    with pytest.raises(TypeError, match="must be declared with"):

        class BadSignature(Signature):
            """Bad signature."""

            question: str  # Missing InputField/OutputField


def test_make_signature() -> None:
    """Test dynamic signature creation."""
    QA = make_signature(
        {"question": str},
        {"answer": str},
        "Answer questions",
    )

    assert QA.get_instructions() == "Answer questions"
    assert "question" in QA.get_input_fields()
    assert "answer" in QA.get_output_fields()


def test_signature_with_multiple_fields() -> None:
    """Test signature with multiple input and output fields."""

    class ComplexSignature(Signature):
        """Complex task."""

        input1: str = InputField()
        input2: int = InputField()
        output1: str = OutputField()
        output2: list[str] = OutputField()

    input_fields = ComplexSignature.get_input_fields()
    output_fields = ComplexSignature.get_output_fields()

    assert len(input_fields) == 2
    assert len(output_fields) == 2
    assert "input1" in input_fields
    assert "input2" in input_fields
    assert "output1" in output_fields
    assert "output2" in output_fields
