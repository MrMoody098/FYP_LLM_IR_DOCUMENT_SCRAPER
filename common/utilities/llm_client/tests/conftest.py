"""Fixtures for OpenAIClient unit tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.utilities.llm_client.open_ai_client import OpenAIClient


@pytest.fixture
def mock_response():
    """Provide a mock OpenAI Responses API response."""
    resp = MagicMock()
    resp.output_text = "Hello from the model"
    return resp


@pytest.fixture
def client(mock_response):
    """Provide an OpenAIClient with a mocked SDK client."""
    with patch.dict("os.environ", {"API_KEY_OPENAI": "test-key"}):
        with patch("common.utilities.llm_client.open_ai_client.OpenAI") as mock_openai, \
             patch("common.utilities.llm_client.open_ai_client.AsyncOpenAI") as mock_async_openai:
            mock_sdk = MagicMock()
            mock_sdk.responses.create.return_value = mock_response
            mock_openai.return_value = mock_sdk
            
            # Also mock the async client
            mock_async_sdk = MagicMock()
            mock_async_sdk.responses.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_async_sdk
            
            yield OpenAIClient(model="test-model")
