"""Tests for the Eden FastAPI server."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------


def _fake_stream(*events):
    """Return a chat_stream function that yields the given event dicts."""

    def _chat_stream(message, thread_id="default"):  # noqa: ARG001
        yield from events

    return _chat_stream


@pytest.fixture()
def stream_client():
    mock_rag = MagicMock()
    mock_rag.chat_stream = _fake_stream(
        {"type": "token", "content": "Hello "},
        {"type": "token", "content": "world"},
        {"type": "done"},
    )
    configure(mock_rag)
    return TestClient(app), mock_rag


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in text.split("\n\n")
        if line.strip().startswith("data: ")
    ]


def test_chat_stream_returns_503_when_rag_not_configured(client):
    response = client.post("/chat/stream", json={"message": "Hello"})
    assert response.status_code == 503


def test_chat_stream_content_type(stream_client):
    client, _ = stream_client
    response = client.post("/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_chat_stream_yields_token_events(stream_client):
    client, _ = stream_client
    response = client.post("/chat/stream", json={"message": "Hello"})
    events = _parse_sse(response.text)
    token_events = [e for e in events if e["type"] == "token"]
    assert token_events == [
        {"type": "token", "content": "Hello "},
        {"type": "token", "content": "world"},
    ]


def test_chat_stream_ends_with_done(stream_client):
    client, _ = stream_client
    response = client.post("/chat/stream", json={"message": "Hello"})
    events = _parse_sse(response.text)
    assert events[-1] == {"type": "done"}


def test_chat_stream_includes_thinking_event():
    mock_rag = MagicMock()
    mock_rag.chat_stream = _fake_stream(
        {"type": "token", "content": "Answer"},
        {"type": "thinking", "content": "I reasoned hard"},
        {"type": "done"},
    )
    configure(mock_rag)
    client = TestClient(app)
    events = _parse_sse(client.post("/chat/stream", json={"message": "Hi"}).text)
    assert {"type": "thinking", "content": "I reasoned hard"} in events


def test_chat_stream_includes_searching_event():
    mock_rag = MagicMock()
    mock_rag.chat_stream = _fake_stream(
        {"type": "searching", "query": "tomato pests"},
        {"type": "token", "content": "Use neem oil."},
        {"type": "done"},
    )
    configure(mock_rag)
    client = TestClient(app)
    events = _parse_sse(client.post("/chat/stream", json={"message": "Tomato?"}).text)
    assert {"type": "searching", "query": "tomato pests"} in events


def test_chat_stream_passes_thread_id():
    calls = []

    def _stream(message, thread_id="default"):
        calls.append((message, thread_id))
        yield {"type": "done"}

    mock_rag = MagicMock()
    mock_rag.chat_stream = _stream
    configure(mock_rag)
    client = TestClient(app)
    client.post("/chat/stream", json={"message": "Hi", "thread_id": "my-thread"})
    assert calls == [("Hi", "my-thread")]


def test_chat_stream_missing_message_returns_422(stream_client):
    client, _ = stream_client
    response = client.post("/chat/stream", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth
# ---------------------------------------------------------------------------


def test_auth_returns_ok_when_no_password_set(client, monkeypatch):
    monkeypatch.delenv("EDEN_PASSWORD", raising=False)
    response = client.post("/auth")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_auth_returns_ok_with_correct_password(client, monkeypatch):
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post("/auth", headers={"X-Password": "secret"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_auth_returns_401_with_wrong_password(client, monkeypatch):
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post("/auth", headers={"X-Password": "wrong"})
    assert response.status_code == 401


def test_auth_returns_401_with_no_header(client, monkeypatch):
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post("/auth")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Password protection on /chat and /chat/stream
# ---------------------------------------------------------------------------


def test_chat_returns_401_with_wrong_password(configured_client, monkeypatch):
    client, _ = configured_client
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post(
        "/chat", json={"message": "Hello"}, headers={"X-Password": "wrong"}
    )
    assert response.status_code == 401


def test_chat_returns_401_with_no_password_header(configured_client, monkeypatch):
    client, _ = configured_client
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 401


def test_chat_succeeds_with_correct_password(configured_client, monkeypatch):
    client, _ = configured_client
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post(
        "/chat", json={"message": "Hello"}, headers={"X-Password": "secret"}
    )
    assert response.status_code == 200


def test_chat_succeeds_with_no_password_set(configured_client, monkeypatch):
    client, _ = configured_client
    monkeypatch.delenv("EDEN_PASSWORD", raising=False)
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200


def test_chat_stream_returns_401_with_wrong_password(stream_client, monkeypatch):
    client, _ = stream_client
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post(
        "/chat/stream", json={"message": "Hello"}, headers={"X-Password": "wrong"}
    )
    assert response.status_code == 401


def test_chat_stream_succeeds_with_correct_password(stream_client, monkeypatch):
    client, _ = stream_client
    monkeypatch.setenv("EDEN_PASSWORD", "secret")
    response = client.post(
        "/chat/stream", json={"message": "Hello"}, headers={"X-Password": "secret"}
    )
    assert response.status_code == 200
