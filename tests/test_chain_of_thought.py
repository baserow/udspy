"""Tests for ChainOfThought module."""

from unittest.mock import AsyncMock

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from udspy import ChainOfThought, InputField, OutputField, Signature, settings


def test_chain_of_thought_initialization() -> None:
    """Test ChainOfThought module initialization."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    cot = ChainOfThought(QA)

    assert cot.original_signature == QA
    assert hasattr(cot, "predict")


def test_chain_of_thought_adds_reasoning_field() -> None:
    """Test that ChainOfThought adds reasoning to signature."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    cot = ChainOfThought(QA)

    # Check that the extended signature has reasoning
    extended_sig = cot.predict.signature
    output_fields = extended_sig.get_output_fields()

    assert "reasoning" in output_fields
    assert "answer" in output_fields


def test_chain_of_thought_forward() -> None:
    """Test ChainOfThought prediction with reasoning."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Mock streaming response with reasoning and answer
    chunks = [
        ChatCompletionChunk(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            created=1234567890,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        content=(
                            "[[ ## reasoning ## ]]\n"
                            "Let's think step by step. 2+2 is basic addition.\n"
                            "[[ ## answer ## ]]\n"
                            "4"
                        ),
                        role="assistant",
                    ),
                    finish_reason=None,
                )
            ],
        ),
    ]

    async def mock_stream():
        for chunk in chunks:
            yield chunk

    mock_async_client = settings.aclient
    mock_async_client.chat.completions.create = AsyncMock(return_value=mock_stream())

    cot = ChainOfThought(QA)
    result = cot(question="What is 2+2?")

    assert "reasoning" in result
    assert "answer" in result
    assert "step by step" in result.reasoning.lower()
    assert "4" in result.answer


def test_chain_of_thought_preserves_instructions() -> None:
    """Test that ChainOfThought preserves original instructions."""

    class QA(Signature):
        """Answer questions concisely and accurately."""

        question: str = InputField()
        answer: str = OutputField()

    cot = ChainOfThought(QA)
    extended_sig = cot.predict.signature

    # Instructions should be preserved
    assert extended_sig.get_instructions() == QA.get_instructions()


def test_chain_of_thought_with_custom_description() -> None:
    """Test ChainOfThought with custom reasoning description."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    cot = ChainOfThought(QA, reasoning_description="Detailed thought process")

    extended_sig = cot.predict.signature
    reasoning_field = extended_sig.get_output_fields()["reasoning"]

    assert reasoning_field.description == "Detailed thought process"


def test_chain_of_thought_with_multiple_outputs() -> None:
    """Test ChainOfThought with multiple output fields."""

    class ComplexTask(Signature):
        """Complex task with multiple outputs."""

        query: str = InputField()
        analysis: str = OutputField()
        recommendation: str = OutputField()

    cot = ChainOfThought(ComplexTask)
    extended_sig = cot.predict.signature

    output_fields = list(extended_sig.get_output_fields().keys())

    # Reasoning should be first, then original outputs
    assert output_fields[0] == "reasoning"
    assert "analysis" in output_fields
    assert "recommendation" in output_fields


def test_chain_of_thought_with_kwargs() -> None:
    """Test ChainOfThought with custom model parameters."""

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    cot = ChainOfThought(QA, temperature=0.0, max_tokens=100)

    assert cot.predict.kwargs["temperature"] == 0.0
    assert cot.predict.kwargs["max_tokens"] == 100
