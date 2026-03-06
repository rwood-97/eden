"""Tests for openai_client and azure_client factory functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# openai_client.make_client
# ---------------------------------------------------------------------------


def test_make_client_uses_explicit_args():
    with patch("eden.openai_client.OpenAI") as mock_openai:
        from eden.openai_client import make_client

        make_client(base_url="http://localhost:8000", api_key="sk-test")

        mock_openai.assert_called_once_with(
            base_url="http://localhost:8000", api_key="sk-test"
        )


def test_make_client_reads_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://env-host/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    with patch("eden.openai_client.OpenAI") as mock_openai:
        from eden.openai_client import make_client

        make_client()

        mock_openai.assert_called_once_with(
            base_url="http://env-host/v1", api_key="sk-env"
        )


def test_make_client_defaults_api_key_to_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("eden.openai_client.OpenAI") as mock_openai:
        from eden.openai_client import make_client

        make_client()

        assert mock_openai.call_args.kwargs["api_key"] == "EMPTY"


def test_make_client_falls_back_to_vllm_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with patch("eden.openai_client.OpenAI") as mock_openai:
        from eden.openai_client import make_client

        make_client()

        assert mock_openai.call_args.kwargs["base_url"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# openai_client.get_tool_response
# ---------------------------------------------------------------------------

_TOOL = {
    "type": "function",
    "function": {
        "name": "save_result",
        "description": "Save result",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def test_get_tool_response_returns_arguments():
    from eden.openai_client import get_tool_response

    mock_client = MagicMock()
    mock_tc = MagicMock()
    mock_tc.function.arguments = '{"key": "value"}'
    mock_client.chat.completions.create.return_value.choices[0].message.tool_calls = [
        mock_tc
    ]

    result = get_tool_response(mock_client, "prompt", "model-x", _TOOL)

    assert result == '{"key": "value"}'


def test_get_tool_response_passes_correct_tool_choice():
    from eden.openai_client import get_tool_response

    mock_client = MagicMock()
    mock_tc = MagicMock()
    mock_tc.function.arguments = "{}"
    mock_client.chat.completions.create.return_value.choices[0].message.tool_calls = [
        mock_tc
    ]

    get_tool_response(mock_client, "my prompt", "gpt-x", _TOOL)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "save_result"},
    }
    assert call_kwargs["messages"] == [{"role": "user", "content": "my prompt"}]


def test_get_tool_response_returns_none_on_exception():
    from eden.openai_client import get_tool_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    result = get_tool_response(mock_client, "prompt", "model", _TOOL)

    assert result is None


# ---------------------------------------------------------------------------
# azure_client.make_azure_client
# ---------------------------------------------------------------------------


def test_make_azure_client_uses_explicit_args():
    with patch("eden.azure_client.AzureOpenAI") as mock_azure:
        from eden.azure_client import make_azure_client

        make_azure_client(
            base_url="https://myresource.openai.azure.com/openai/deployments/",
            api_key="azure-key",
        )

        mock_azure.assert_called_once_with(
            base_url="https://myresource.openai.azure.com/openai/deployments/",
            api_key="azure-key",
            api_version="2025-01-01-preview",
        )


def test_make_azure_client_appends_model_to_base_url():
    with patch("eden.azure_client.AzureOpenAI") as mock_azure:
        from eden.azure_client import make_azure_client

        make_azure_client(
            base_url="https://myresource.openai.azure.com/openai/deployments/",
            api_key="key",
            model="gpt-4o",
        )

        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["base_url"] == (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o/"
        )


def test_make_azure_client_strips_trailing_slash_before_appending_model():
    with patch("eden.azure_client.AzureOpenAI") as mock_azure:
        from eden.azure_client import make_azure_client

        make_azure_client(
            base_url="https://myresource.openai.azure.com/openai/deployments",
            api_key="key",
            model="gpt-4o",
        )

        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["base_url"] == (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o/"
        )


def test_make_azure_client_custom_api_version():
    with patch("eden.azure_client.AzureOpenAI") as mock_azure:
        from eden.azure_client import make_azure_client

        make_azure_client(
            base_url="https://example.com/",
            api_key="key",
            api_version="2024-06-01",
        )

        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["api_version"] == "2024-06-01"


def test_make_azure_client_reads_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://env.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-azure-key")

    with patch("eden.azure_client.AzureOpenAI") as mock_azure:
        from eden.azure_client import make_azure_client

        make_azure_client()

        call_kwargs = mock_azure.call_args.kwargs
        assert call_kwargs["base_url"] == "https://env.openai.azure.com/"
        assert call_kwargs["api_key"] == "env-azure-key"


def test_make_azure_client_missing_env_raises(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    import importlib

    import eden.azure_client as mod

    importlib.reload(mod)

    with pytest.raises(KeyError):
        mod.make_azure_client()
