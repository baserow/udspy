"""Tests for retry logic on AdapterParseError."""

from unittest.mock import AsyncMock

import pytest
from conftest import make_mock_response

from udspy import InputField, OutputField, Predict, Signature, settings
from udspy.adapter import ChatAdapter
from udspy.exceptions import AdapterParseError


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


class FailingThenSucceedingAdapter(ChatAdapter):
    """Adapter that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        super().__init__()
        self.fail_count = fail_count
        self.call_count = 0

    def parse_outputs(self, signature, completion):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            # Simulate parse error
            raise AdapterParseError(
                adapter_name="TestAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Simulated failure {self.call_count}",
            )
        # Succeed
        return super().parse_outputs(signature, completion)


class AlwaysFailingAdapter(ChatAdapter):
    """Adapter that always fails."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def parse_outputs(self, signature, completion):
        self.call_count += 1
        raise AdapterParseError(
            adapter_name="TestAdapter",
            signature=signature,
            lm_response=completion,
            message=f"Simulated failure {self.call_count}",
        )


@pytest.mark.asyncio
@pytest.mark.skipif(
    True, reason="Requires live OpenAI API - run manually with valid API key if needed"
)
async def test_nonstreaming_retry_eventually_succeeds():
    """Test that retries eventually succeed when adapter stops failing."""
    # Create predictor with adapter that fails twice then succeeds
    adapter = FailingThenSucceedingAdapter(fail_count=2)
    predictor = Predict(QA, adapter=adapter)

    # This would require a real API call, so skip in CI
    result = await predictor(question="What is 2+2?")

    # Should have called parse 3 times (2 failures + 1 success)
    assert adapter.call_count == 3
    assert "answer" in result


@pytest.mark.asyncio
@pytest.mark.skipif(
    True, reason="Requires live OpenAI API - run manually with valid API key if needed"
)
async def test_nonstreaming_max_retries_respected():
    """Test that retry stops after max attempts."""
    # Create predictor with adapter that always fails
    adapter = AlwaysFailingAdapter()
    predictor = Predict(QA, adapter=adapter)

    # Should fail after 3 attempts
    with pytest.raises(AdapterParseError):
        await predictor(question="What is 2+2?")

    # Should have attempted 3 times
    assert adapter.call_count == 3


def test_retry_decorator_configuration():
    """Test that retry decorators are properly configured on methods."""
    from tenacity import AsyncRetrying

    predictor = Predict(QA)

    # Check that _process_nonstreaming has retry decorator
    assert hasattr(predictor._aforward, "retry")
    assert isinstance(predictor._aforward.retry, AsyncRetrying)

    # Check that _process_streaming has retry decorator
    assert hasattr(predictor._astream, "retry")
    assert isinstance(predictor._astream.retry, AsyncRetrying)


def test_adapter_parse_error_exists():
    """Test that AdapterParseError is properly defined."""
    # Create a test error
    error = AdapterParseError(
        adapter_name="TestAdapter",
        signature=QA,
        lm_response="test response",
        message="test message",
    )

    assert error.adapter_name == "TestAdapter"
    assert error.signature == QA
    assert error.lm_response == "test response"
    assert "TestAdapter" in str(error)
    assert "test response" in str(error)


@pytest.mark.asyncio
async def test_aforward_retries_on_adapter_parse_error():
    """Test that _aforward retries up to 3 times when AdapterParseError is raised."""
    # Create adapter that fails twice then succeeds
    adapter = FailingThenSucceedingAdapter(fail_count=2)
    predictor = Predict(QA, adapter=adapter)

    # Mock the API response
    mock_response = make_mock_response("[[ ## answer ## ]]\n4")
    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    # Execute - should retry and eventually succeed
    result = await predictor.aforward(question="What is 2+2?")

    # Should have called parse_outputs 3 times (2 failures + 1 success)
    assert adapter.call_count == 3
    assert result.answer == "4"


@pytest.mark.asyncio
async def test_aforward_stops_after_max_retries():
    """Test that _aforward stops retrying after 3 attempts."""
    from tenacity import RetryError

    # Create adapter that always fails
    adapter = AlwaysFailingAdapter()
    predictor = Predict(QA, adapter=adapter)

    # Mock the API response
    mock_response = make_mock_response("[[ ## answer ## ]]\n4")
    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    # Should fail after 3 attempts (tenacity wraps in RetryError)
    with pytest.raises(RetryError):
        await predictor.aforward(question="What is 2+2?")

    # Should have attempted exactly 3 times
    assert adapter.call_count == 3


@pytest.mark.asyncio
async def test_astream_retries_on_adapter_parse_error():
    """Test that _astream retries up to 3 times when AdapterParseError is raised."""
    # Create adapter that fails twice then succeeds
    adapter = FailingThenSucceedingAdapter(fail_count=2)
    predictor = Predict(QA, adapter=adapter)

    # Mock streaming response - return a fresh generator each time
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    def make_chunks():
        return [
            ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content="[[ ## answer ## ]]\n4"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[Choice(index=0, delta=ChoiceDelta(content=""), finish_reason="stop")],
            ),
        ]

    async def mock_stream():
        for chunk in make_chunks():
            yield chunk

    # Return a new stream on each call
    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: mock_stream())

    # Execute - should retry and eventually succeed
    from udspy.streaming import Prediction

    final_result = None
    async for event in predictor.astream(question="What is 2+2?"):
        if isinstance(event, Prediction):
            final_result = event

    # Should have called parse_outputs 3 times (2 failures + 1 success)
    assert adapter.call_count == 3
    assert final_result is not None
    assert final_result.answer == "4"


@pytest.mark.asyncio
async def test_astream_stops_after_max_retries():
    """Test that _astream stops retrying after 3 attempts."""
    from tenacity import RetryError

    # Create adapter that always fails
    adapter = AlwaysFailingAdapter()
    predictor = Predict(QA, adapter=adapter)

    # Mock streaming response - return a fresh generator each time
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    def make_chunks():
        return [
            ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content="[[ ## answer ## ]]\n4"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[Choice(index=0, delta=ChoiceDelta(content=""), finish_reason="stop")],
            ),
        ]

    async def mock_stream():
        for chunk in make_chunks():
            yield chunk

    # Return a new stream on each call
    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: mock_stream())

    # Should fail after 3 attempts (tenacity wraps in RetryError)
    with pytest.raises(RetryError):
        async for _ in predictor.astream(question="What is 2+2?"):
            pass

    # Should have attempted exactly 3 times
    assert adapter.call_count == 3
