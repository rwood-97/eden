"""Tests for the RAG CLI backend selection (openai / azure / ollama)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from eden.rag.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared patch helpers
# ---------------------------------------------------------------------------


def _retriever_patches():
    """Patch out all heavy RAG deps so the CLI can run without ML models."""
    mock_collection = MagicMock()
    mock_splitter = MagicMock()

    return (
        patch(
            "eden.rag.build_retriever.get_retriever",
            return_value=(mock_collection, mock_splitter),
        ),
        patch("eden.rag.rag.RAG"),
        patch("eden.rag.build_retriever.RetrieverConfig"),
        patch("eden.rag.build_retriever.DEFAULT_RETRIEVER_CONFIG"),
    )


# ---------------------------------------------------------------------------
# build-index
# ---------------------------------------------------------------------------


def test_build_index_no_backend_required(tmp_path):
    source_file = tmp_path / "advice.jsonl"
    source_file.write_text('{"url": "http://x.com", "title": "T", "sections": []}\n')
    persist_dir = tmp_path / "chroma"

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with p_retriever, p_rag, p_config, p_default, patch("eden.rag.rag.index_documents"):
        result = runner.invoke(
            app,
            [
                "build-index",
                "--source-file",
                str(source_file),
                "--source-type",
                "advice",
                "--persist-dir",
                str(persist_dir),
            ],
        )

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# chat — backend selection
# ---------------------------------------------------------------------------


def test_chat_invalid_backend(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    result = runner.invoke(
        app,
        ["chat", "--persist-dir", str(persist_dir), "--backend", "invalid"],
        input="quit\n",
    )

    assert result.exit_code != 0


def test_chat_ollama_uses_make_client(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client") as mock_make_client,
        patch("eden.azure_client.make_azure_client"),
    ):
        mock_make_client.return_value = MagicMock()

        runner.invoke(
            app,
            ["chat", "--persist-dir", str(persist_dir), "--backend", "ollama"],
            input="quit\n",
        )

        mock_make_client.assert_called_once()
        call_kwargs = mock_make_client.call_args.kwargs
        assert "11434" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "ollama"


def test_chat_ollama_respects_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remotehost:11434/v1")
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client") as mock_make_client,
        patch("eden.azure_client.make_azure_client"),
    ):
        mock_make_client.return_value = MagicMock()

        runner.invoke(
            app,
            ["chat", "--persist-dir", str(persist_dir), "--backend", "ollama"],
            input="quit\n",
        )

        call_kwargs = mock_make_client.call_args.kwargs
        assert call_kwargs["base_url"] == "http://remotehost:11434/v1"


def test_chat_ollama_does_not_call_azure_client(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client"),
        patch("eden.azure_client.make_azure_client") as mock_azure,
    ):
        runner.invoke(
            app,
            ["chat", "--persist-dir", str(persist_dir), "--backend", "ollama"],
            input="quit\n",
        )

        mock_azure.assert_not_called()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_invalid_backend(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    result = runner.invoke(
        app,
        ["serve", "--persist-dir", str(persist_dir), "--backend", "invalid"],
    )

    assert result.exit_code != 0


def test_serve_missing_persist_dir(tmp_path):
    result = runner.invoke(
        app,
        ["serve", "--persist-dir", str(tmp_path / "nonexistent")],
    )

    assert result.exit_code != 0


def test_serve_ollama_uses_make_client(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client") as mock_make_client,
        patch("eden.azure_client.make_azure_client"),
        patch("eden.rag.server.configure"),
        patch("uvicorn.run"),
    ):
        mock_make_client.return_value = MagicMock()

        runner.invoke(
            app,
            ["serve", "--persist-dir", str(persist_dir), "--backend", "ollama"],
        )

        mock_make_client.assert_called_once()
        call_kwargs = mock_make_client.call_args.kwargs
        assert "11434" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "ollama"


def test_serve_ollama_respects_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remotehost:11434/v1")
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client") as mock_make_client,
        patch("eden.azure_client.make_azure_client"),
        patch("eden.rag.server.configure"),
        patch("uvicorn.run"),
    ):
        mock_make_client.return_value = MagicMock()

        runner.invoke(
            app,
            ["serve", "--persist-dir", str(persist_dir), "--backend", "ollama"],
        )

        call_kwargs = mock_make_client.call_args.kwargs
        assert call_kwargs["base_url"] == "http://remotehost:11434/v1"


def test_serve_starts_uvicorn(tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    p_retriever, p_rag, p_config, p_default = _retriever_patches()
    with (
        p_retriever,
        p_rag,
        p_config,
        p_default,
        patch("eden.openai_client.make_client", return_value=MagicMock()),
        patch("eden.azure_client.make_azure_client"),
        patch("eden.rag.server.configure"),
        patch("uvicorn.run") as mock_uvicorn,
    ):
        runner.invoke(
            app,
            ["serve", "--persist-dir", str(persist_dir), "--port", "9000"],
        )

        mock_uvicorn.assert_called_once()
        assert mock_uvicorn.call_args.kwargs["port"] == 9000
