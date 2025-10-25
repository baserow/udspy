"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI

import udspy


@pytest.fixture(autouse=True)
def configure_client() -> None:
    """Configure a mock async OpenAI client for testing."""
    # Use mock async client to avoid actual API calls
    # (Sync wrappers use asyncio.run() which works with async client)
    mock_aclient = MagicMock(spec=AsyncOpenAI)

    udspy.settings.configure(
        aclient=mock_aclient,
        model="gpt-4o-mini",
    )


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
