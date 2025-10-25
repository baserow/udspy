"""Tests for modules accepting string signatures."""

from udspy import ChainOfThought, Predict, Signature


def test_predict_accepts_string_signature():
    """Test that Predict accepts string signature."""
    predictor = Predict("question -> answer")

    # Verify signature was created correctly
    assert hasattr(predictor, "signature")
    assert isinstance(predictor.signature, type)
    assert issubclass(predictor.signature, Signature)

    inputs = predictor.signature.get_input_fields()
    outputs = predictor.signature.get_output_fields()

    assert "question" in inputs
    assert "answer" in outputs


def test_predict_string_vs_class_equivalent():
    """Test that string and class signatures work the same."""
    # String signature
    pred_string = Predict("question -> answer")

    # Both should have correct field names
    string_inputs = set(pred_string.signature.get_input_fields().keys())
    string_outputs = set(pred_string.signature.get_output_fields().keys())

    assert string_inputs == {"question"}
    assert string_outputs == {"answer"}


def test_chain_of_thought_accepts_string_signature():
    """Test that ChainOfThought accepts string signature."""
    cot = ChainOfThought("question -> answer")

    # Should have reasoning field added
    outputs = cot.predict.signature.get_output_fields()

    assert "reasoning" in outputs
    assert "answer" in outputs


def test_chain_of_thought_multiple_fields():
    """Test ChainOfThought with multiple inputs/outputs."""
    cot = ChainOfThought("context, question -> summary, answer")

    inputs = cot.predict.signature.get_input_fields()
    outputs = cot.predict.signature.get_output_fields()

    assert "context" in inputs
    assert "question" in inputs
    assert "reasoning" in outputs
    assert "summary" in outputs
    assert "answer" in outputs
