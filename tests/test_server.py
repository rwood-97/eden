"""Tests for the Eden FastAPI server."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from eden.rag.rag import ChatResult
from eden.rag.server import app, configure


@pytest.fixture(autouse=True)
def _reset_rag():
    """Ensure _rag is reset to None between tests."""
    import eden.rag.server as srv

    original = srv._rag
    yield
    srv._rag = original


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def configured_client():
    mock_rag = MagicMock()
    mock_rag.chat.return_value = ChatResult(
        reply="Tomatoes need full sun and regular watering.", thinking=""
    )
    configure(mock_rag)
    return TestClient(app), mock_rag


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_index_contains_chat_ui(client):
    response = client.get("/")
    assert "Eden" in response.text
    assert "chat" in response.text.lower()


# ---------------------------------------------------------------------------
# POST /chat — unconfigured
# ---------------------------------------------------------------------------


def test_chat_returns_503_when_rag_not_configured(client):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /chat — configured
# ---------------------------------------------------------------------------


def test_chat_returns_reply(configured_client):
    client, mock_rag = configured_client
    response = client.post("/chat", json={"message": "How do I grow tomatoes?"})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Tomatoes need full sun and regular watering."
    assert data["thread_id"] == "default"


def test_chat_passes_message_to_rag(configured_client):
    client, mock_rag = configured_client
    client.post("/chat", json={"message": "What is aphid?"})
    mock_rag.chat.assert_called_once_with("What is aphid?", thread_id="default")


def test_chat_passes_custom_thread_id(configured_client):
    client, mock_rag = configured_client
    client.post("/chat", json={"message": "Hello", "thread_id": "session-abc"})
    mock_rag.chat.assert_called_once_with("Hello", thread_id="session-abc")


def test_chat_thread_id_echoed_in_response(configured_client):
    client, mock_rag = configured_client
    response = client.post("/chat", json={"message": "Hello", "thread_id": "my-thread"})
    assert response.json()["thread_id"] == "my-thread"


def test_chat_thinking_included_in_response(configured_client):
    client, mock_rag = configured_client
    mock_rag.chat.return_value = ChatResult(
        reply="Use insecticidal soap.", thinking="I should search for aphid remedies."
    )
    response = client.post("/chat", json={"message": "How do I control aphids?"})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Use insecticidal soap."
    assert data["thinking"] == "I should search for aphid remedies."


def test_chat_thinking_empty_when_no_reasoning(configured_client):
    client, _ = configured_client
    response = client.post("/chat", json={"message": "Hello"})
    assert response.json()["thinking"] == ""


def test_chat_missing_message_returns_422(configured_client):
    client, _ = configured_client
    response = client.post("/chat", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


def test_configure_sets_rag():
    import eden.rag.server as srv

    mock_rag = MagicMock()
    configure(mock_rag)
    assert srv._rag is mock_rag


def test_configure_can_be_called_multiple_times():
    import eden.rag.server as srv

    rag_a = MagicMock()
    rag_b = MagicMock()
    configure(rag_a)
    configure(rag_b)
    assert srv._rag is rag_b
