"""Tests for Signature.from_string() method."""

import pytest

from udspy import Predict, Signature


def test_simple_signature():
    """Test simple 'question -> answer' format."""
    QA = Signature.from_string("question -> answer", "Answer questions")

    # Check fields
    inputs = QA.get_input_fields()
    outputs = QA.get_output_fields()

    assert "question" in inputs
    assert "answer" in outputs
    assert len(inputs) == 1
    assert len(outputs) == 1

    # Check instructions
    assert QA.get_instructions() == "Answer questions"


def test_multiple_inputs():
    """Test multiple input fields."""
    Sig = Signature.from_string("context, question -> answer")

    inputs = Sig.get_input_fields()
    outputs = Sig.get_output_fields()

    assert "context" in inputs
    assert "question" in inputs
    assert "answer" in outputs
    assert len(inputs) == 2
    assert len(outputs) == 1


def test_multiple_outputs():
    """Test multiple output fields."""
    Sig = Signature.from_string("text -> summary, keywords")

    inputs = Sig.get_input_fields()
    outputs = Sig.get_output_fields()

    assert "text" in inputs
    assert "summary" in outputs
    assert "keywords" in outputs
    assert len(inputs) == 1
    assert len(outputs) == 2


def test_multiple_inputs_and_outputs():
    """Test multiple inputs and outputs."""
    Sig = Signature.from_string("doc, style -> summary, keywords, sentiment")

    inputs = Sig.get_input_fields()
    outputs = Sig.get_output_fields()

    assert len(inputs) == 2
    assert len(outputs) == 3
    assert "doc" in inputs
    assert "style" in inputs
    assert "summary" in outputs
    assert "keywords" in outputs
    assert "sentiment" in outputs


def test_whitespace_handling():
    """Test that whitespace is properly handled."""
    Sig = Signature.from_string("  input1  ,  input2  ->  output1  ,  output2  ")

    inputs = Sig.get_input_fields()
    outputs = Sig.get_output_fields()

    assert "input1" in inputs
    assert "input2" in inputs
    assert "output1" in outputs
    assert "output2" in outputs


def test_no_arrow_raises_error():
    """Test that missing arrow raises ValueError."""
    with pytest.raises(ValueError, match="Invalid signature format"):
        Signature.from_string("question answer")


def test_multiple_arrows_raises_error():
    """Test that multiple arrows raise ValueError."""
    with pytest.raises(ValueError, match="exactly one '->' separator"):
        Signature.from_string("question -> answer -> extra")


def test_empty_inputs_raises_error():
    """Test that empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="at least one input field"):
        Signature.from_string("-> answer")


def test_empty_outputs_raises_error():
    """Test that empty outputs raise ValueError."""
    with pytest.raises(ValueError, match="at least one output field"):
        Signature.from_string("question ->")


def test_works_with_predict():
    """Test that Signature.from_string works with Predict module."""
    QA = Signature.from_string("question -> answer", "Answer questions")
    # Just test that we can create a predictor with this signature
    predictor = Predict(QA)
    assert predictor is not None


def test_all_fields_are_strings():
    """Test that all fields default to str type."""
    Sig = Signature.from_string("input1, input2 -> output1, output2")

    # Create an instance to check types
    instance = Sig(input1="test", input2="test2", output1="out1", output2="out2")
    assert isinstance(instance.input1, str)
    assert isinstance(instance.input2, str)
    assert isinstance(instance.output1, str)
    assert isinstance(instance.output2, str)


def test_dspy_compatibility_example():
    """Test DSPy-style usage example."""
    # This mimics DSPy's signature creation
    QA = Signature.from_string("question -> answer")

    inputs = QA.get_input_fields()
    outputs = QA.get_output_fields()

    assert "question" in inputs
    assert "answer" in outputs
