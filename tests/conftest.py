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
