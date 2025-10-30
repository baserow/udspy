"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

import udspy
from udspy.exceptions import AdapterParseError
from udspy.module.predict import Predict


@pytest.fixture(autouse=True)
def fast_retry():
    """Patch retry decorators to use no wait time for fast tests."""
    # Create a fast retry decorator with no wait
    fast_retry_decorator = retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_none(),  # No wait between retries
    )

    # Patch both _aforward and _astream methods
    with patch(
        "udspy.module.predict.Predict._aforward",
        new=fast_retry_decorator(Predict._aforward.__wrapped__),
    ):
        with patch(
            "udspy.module.predict.Predict._astream",
            new=fast_retry_decorator(Predict._astream.__wrapped__),
        ):
            yield


@pytest.fixture(autouse=True)
def configure_client() -> None:
    """Configure a mock LM for testing."""
    # Use mock async client to avoid actual API calls
    # (Sync wrappers use asyncio.run() which works with async client)
    from udspy.lm import OpenAILM

    mock_aclient = MagicMock(spec=AsyncOpenAI)
    mock_lm = OpenAILM(mock_aclient, default_model="gpt-4o-mini")

    udspy.settings.configure(lm=mock_lm)


@pytest.fixture
def api_key() -> str:
    """Get OpenAI API key from environment (for integration tests)."""
    return os.getenv("OPENAI_API_KEY", "sk-test-key")


def make_mock_response(content: str, tool_calls: list | None = None, streaming: bool = False):
    """Create OpenAI API mock response.

    Returns streaming or non-streaming response based on streaming parameter.
    This is the ONLY thing tests should mock - the LLM API response.
    """
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    if streaming:

        async def stream():
            yield ChatCompletionChunk(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion.chunk",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=content, tool_calls=tool_calls),
                        finish_reason=None,
                    )
                ],
            )

        return stream()
    else:
        return ChatCompletion(
            id="test",
            model="gpt-4o-mini",
            object="chat.completion",
            created=1234567890,
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason="stop",
                )
            ],
        )
