"""Tests for the RAG class and index_documents function."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from eden.rag.rag import RAG, index_documents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_message(content: str | None, tool_calls=None) -> MagicMock:
    """Build a mock OpenAI ChatCompletionMessage."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


def _make_openai_response(content: str | None, tool_calls=None) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message = _make_openai_message(content, tool_calls)
    return resp


def _make_tool_call(query: str, call_id: str = "call_abc") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = "search_gardening_knowledge"
    tc.function.arguments = json.dumps({"query": query})
    return tc


def _make_rag(model: str = "gpt-4o-mini") -> RAG:
    """Create a RAG instance with fully mocked chromadb dependencies."""
    collection = MagicMock()
    client = MagicMock()

    return RAG(
        collection=collection,
        client=client,
        model=model,
    )


# ---------------------------------------------------------------------------
# chat() — basic behaviour
# ---------------------------------------------------------------------------


def test_chat_returns_string():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response(
        "Roses need full sun and well-drained soil."
    )

    result = rag.chat("How do I grow roses?")

    assert isinstance(result, str)
    assert result == "Roses need full sun and well-drained soil."


def test_chat_empty_content_returns_empty_string():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response(None)

    result = rag.chat("Hello")

    assert result == ""


def test_chat_calls_llm_with_user_message():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response("Answer.")

    rag.chat("What is aphid?")

    call_args = rag.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_messages = [m for m in messages if m["role"] == "user"]
    assert any("aphid" in m["content"] for m in user_messages)


def test_chat_with_tool_call_triggers_search_then_responds():
    rag = _make_rag()
    tc = _make_tool_call("aphid control")

    rag.client.chat.completions.create.side_effect = [
        _make_openai_response(None, tool_calls=[tc]),
        _make_openai_response("Use insecticidal soap."),
    ]

    with patch.object(
        rag, "_search", return_value="Aphid content from KB"
    ) as mock_search:
        result = rag.chat("How do I control aphids?")

    mock_search.assert_called_once_with("aphid control")
    assert result == "Use insecticidal soap."
    assert rag.client.chat.completions.create.call_count == 2


def test_chat_tool_call_result_added_to_history():
    rag = _make_rag()
    tc = _make_tool_call("roses", call_id="call_xyz")

    rag.client.chat.completions.create.side_effect = [
        _make_openai_response(None, tool_calls=[tc]),
        _make_openai_response("Roses grow well in sun."),
    ]

    with patch.object(rag, "_search", return_value="Rose knowledge"):
        rag.chat("Tell me about roses")

    # Second LLM call should include a tool result message
    second_call_messages = rag.client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_xyz"
    assert "Rose knowledge" in tool_messages[0]["content"]


# ---------------------------------------------------------------------------
# chat() — conversation history
# ---------------------------------------------------------------------------


def test_chat_maintains_history_across_turns():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response("Answer.")

    rag.chat("First question", thread_id="t1")
    rag.chat("Second question", thread_id="t1")

    # Second call messages should include both user turns
    messages = rag.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    user_contents = [m["content"] for m in messages if m["role"] == "user"]
    assert "First question" in user_contents
    assert "Second question" in user_contents


def test_chat_separate_threads_are_isolated():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response("Answer.")

    rag.chat("Thread A message", thread_id="a")
    rag.chat("Thread B message", thread_id="b")

    # Thread B's call should not contain thread A's message
    b_messages = rag.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    all_content = " ".join(m.get("content", "") or "" for m in b_messages)
    assert "Thread A message" not in all_content


def test_chat_system_prompt_added_on_first_message():
    rag = _make_rag()
    rag.client.chat.completions.create.return_value = _make_openai_response("Hi.")

    rag.chat("Hello", thread_id="new_thread")

    messages = rag.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "gardening" in messages[0]["content"].lower()


# ---------------------------------------------------------------------------
# index_documents()
# ---------------------------------------------------------------------------


def _make_indexer():
    """Create mocked collection and text_splitter for index_documents tests."""
    collection = MagicMock()
    text_splitter = MagicMock()
    text_splitter.split_documents.side_effect = lambda docs: docs
    return collection, text_splitter


def test_index_documents_advice_records():
    collection, text_splitter = _make_indexer()
    records = [
        {
            "url": "https://www.rhs.org.uk/grow-your-own/tomatoes",
            "title": "Growing Tomatoes",
            "description": "A guide.",
            "sections": [{"heading": "Planting", "content": "Plant after frost."}],
        }
    ]

    index_documents(collection, text_splitter, records, source_type="advice")

    collection.add.assert_called_once()


def test_index_documents_skips_empty_records():
    collection, text_splitter = _make_indexer()
    records = [
        {
            "url": "https://www.rhs.org.uk/plants/1",
            "commonName": "",
            "botanicalNameUnFormatted": "",
            "cultivation": "",
            "pruning": "",
            "propagation": "",
        },
    ]

    index_documents(collection, text_splitter, records, source_type="plants")

    collection.add.assert_not_called()


def test_index_documents_plants_records():
    collection, text_splitter = _make_indexer()
    records = [
        {
            "url": "https://www.rhs.org.uk/plants/1234",
            "commonName": "Rose",
            "cultivation": "Plant in full sun.",
            "pruning": "Prune in late winter.",
            "propagation": "",
        }
    ]

    index_documents(collection, text_splitter, records, source_type="plants")

    collection.add.assert_called_once()


def test_index_documents_multiple_records_batched():
    collection, text_splitter = _make_indexer()
    records = [
        {
            "url": f"https://www.rhs.org.uk/advice/{i}",
            "title": f"Article {i}",
            "sections": [{"heading": "Section", "content": f"Content {i}"}],
        }
        for i in range(5)
    ]

    index_documents(collection, text_splitter, records, source_type="advice")

    # All docs added in one call (after splitting)
    collection.add.assert_called_once()


# ---------------------------------------------------------------------------
# _search() — internal helper
# ---------------------------------------------------------------------------


def test_search_returns_string():
    rag = _make_rag()
    rag.collection.query.return_value = {
        "documents": [["Roses need sun."]],
        "metadatas": [[{"source": "https://www.rhs.org.uk/roses", "title": "Roses"}]],
    }

    result = rag._search("roses")

    assert isinstance(result, str)
    assert "Roses need sun." in result


def test_search_no_results_returns_fallback():
    rag = _make_rag()
    rag.collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

    result = rag._search("obscure topic")

    assert "No relevant information" in result
