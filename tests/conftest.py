"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI, OpenAI

import udspy


@pytest.fixture(autouse=True)
def configure_client() -> None:
    """Configure a mock OpenAI client for testing."""
    # Use mock client to avoid actual API calls
    mock_client = MagicMock(spec=OpenAI)
    mock_async_client = MagicMock(spec=AsyncOpenAI)

    udspy.settings.configure(
        client=mock_client,
        async_client=mock_async_client,
        model="gpt-4o-mini",
    )


@pytest.fixture
def api_key() -> str:
    """Get OpenAI API key from environment (for integration tests)."""
    return os.getenv("OPENAI_API_KEY", "sk-test-key")
