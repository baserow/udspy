"""Tests for module abstraction."""

from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from udspy import InputField, OutputField, Predict, Prediction, Signature, settings


def test_prediction_attribute_access() -> None:
    """Test Prediction allows both dict and attribute access."""
    pred = Prediction(answer="Paris", reasoning="Capital of France")

    assert pred["answer"] == "Paris"
    assert pred.answer == "Paris"
    assert pred["reasoning"] == "Capital of France"
    assert pred.reasoning == "Capital of France"


def test_prediction_missing_attribute() -> None:
    """Test accessing missing attributes raises AttributeError."""
    pred = Prediction(answer="Paris")

    with pytest.raises(AttributeError):
        _ = pred.nonexistent


def test_predict_initialization() -> None:
    """Test Predict module initialization."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    predictor = Predict(QA)

    assert predictor.signature == QA
    assert predictor.model == "gpt-4o-mini"
    assert isinstance(predictor.tools, list)


def test_predict_forward() -> None:
    """Test basic prediction."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Mock the OpenAI response
    mock_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\nParis",
                ),
                finish_reason="stop",
            )
        ],
    )

    mock_client = settings.client
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    predictor = Predict(QA)
    result = predictor(question="What is the capital of France?")

    assert isinstance(result, Prediction)
    assert "answer" in result
    assert result.answer == "Paris"


def test_predict_missing_input() -> None:
    """Test prediction fails with missing required input."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    predictor = Predict(QA)

    with pytest.raises(ValueError, match="Missing required input field: question"):
        predictor()


def test_predict_with_custom_kwargs() -> None:
    """Test Predict with custom model parameters."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    predictor = Predict(QA, temperature=0.7, max_tokens=100)

    assert predictor.kwargs["temperature"] == 0.7
    assert predictor.kwargs["max_tokens"] == 100
