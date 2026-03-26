"""Unit tests for OpenAIClient."""

from unittest.mock import patch

import pytest

from common.utilities.llm_client.open_ai_client import OpenAIClient


def test_missing_api_key_raises():
    """Test that ValueError is raised when API_KEY_OPENAI is not set."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API_KEY_OPENAI"):
            OpenAIClient()


def test_default_model():
    """Test that the default model is gpt-4.1 when none is specified."""
    with patch.dict("os.environ", {"API_KEY_OPENAI": "test-key"}):
        with patch("common.utilities.llm_client.open_ai_client.OpenAI"), \
             patch("common.utilities.llm_client.open_ai_client.AsyncOpenAI"):
            c = OpenAIClient()
            assert c.model == "gpt-4.1"


def test_chat_returns_output_text(client, mock_response):
    """Test that chat returns the model's output text."""
    mock_response.output_text = "test response"
    result = client.chat("system", "user")
    assert result == "test response"


def test_chat_returns_empty_string_on_none(client, mock_response):
    """Test that chat returns an empty string when output_text is None."""
    mock_response.output_text = None
    result = client.chat("system", "user")
    assert result == ""


def test_chat_json_parses_json_response(client, mock_response):
    """Test that chat_json parses a valid JSON string into a dict."""
    mock_response.output_text = '{"key": "value"}'
    result = client.chat_json("system", "user")
    assert result == {"key": "value"}


def test_chat_json_uses_json_object_format_by_default(client, mock_response):
    """Test that chat_json defaults to json_object format when no schema is provided."""
    mock_response.output_text = '{"ok": true}'
    client.chat_json("system", "user")

    call_kwargs = client._client.responses.create.call_args.kwargs
    assert call_kwargs["text"] == {"format": {"type": "json_object"}}
    assert "tools" not in call_kwargs


def test_chat_json_passes_tools(client, mock_response):
    """Test that tools are forwarded to the API call."""
    mock_response.output_text = '{"ok": true}'
    tools = [{"type": "web_search"}]
    client.chat_json("system", "user", tools=tools)

    call_kwargs = client._client.responses.create.call_args.kwargs
    assert call_kwargs["tools"] == tools


def test_chat_json_uses_json_schema_when_response_schema_provided(client, mock_response):
    """Test that providing a response_schema switches format to json_schema with strict mode."""
    mock_response.output_text = '{"ticker": "NVDA"}'
    schema = {
        "name": "test_schema",
        "schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    }
    client.chat_json("system", "user", response_schema=schema)

    call_kwargs = client._client.responses.create.call_args.kwargs
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert call_kwargs["text"]["format"]["name"] == "test_schema"
    assert call_kwargs["text"]["format"]["strict"] is True


def test_chat_json_schema_defaults_name_to_response(client, mock_response):
    """Test that a schema without a name key defaults to 'response'."""
    mock_response.output_text = '{"ok": true}'
    schema = {
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    client.chat_json("system", "user", response_schema=schema)

    call_kwargs = client._client.responses.create.call_args.kwargs
    assert call_kwargs["text"]["format"]["name"] == "response"


def test_chat_json_raises_on_empty_response(client, mock_response):
    """Test that ValueError is raised when the API returns an empty response."""
    mock_response.output_text = None
    with pytest.raises(ValueError, match="empty response"):
        client.chat_json("system", "user")


def test_chat_json_raises_on_invalid_json(client, mock_response):
    """Test that ValueError is raised when the API returns non-JSON text."""
    mock_response.output_text = "not json at all"
    with pytest.raises(ValueError, match="valid JSON"):
        client.chat_json("system", "user")


# --- Async variants ---


async def test_chat_async_returns_output_text(client, mock_response):
    """Test that chat_async returns the model's output text."""
    mock_response.output_text = "async response"
    result = await client.chat_async("system", "user")
    assert result == "async response"


async def test_chat_async_returns_empty_string_on_none(client, mock_response):
    """Test that chat_async returns an empty string when output_text is None."""
    mock_response.output_text = None
    result = await client.chat_async("system", "user")
    assert result == ""


async def test_chat_json_async_parses_json_response(client, mock_response):
    """Test that chat_json_async parses a valid JSON string into a dict."""
    mock_response.output_text = '{"key": "value"}'
    result = await client.chat_json_async("system", "user")
    assert result == {"key": "value"}


async def test_chat_json_async_raises_on_empty_response(client, mock_response):
    """Test that ValueError is raised when the async API returns an empty response."""
    mock_response.output_text = None
    with pytest.raises(ValueError, match="empty response"):
        await client.chat_json_async("system", "user")


async def test_chat_json_async_raises_on_invalid_json(client, mock_response):
    """Test that ValueError is raised when the async API returns non-JSON text."""
    mock_response.output_text = "not json at all"
    with pytest.raises(ValueError, match="valid JSON"):
        await client.chat_json_async("system", "user")
