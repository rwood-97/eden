import json
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import chromadb
from openai import OpenAI

from eden.data_utils import flatten_record as _flatten_record
from eden.data_utils import get_title as _get_title
from eden.rag.build_index import TokenTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Result of a RAG chat turn."""

    reply: str
    thinking: str = field(default="")


def _extract_thinking(msg: Any) -> tuple[str, str]:
    """Extract a reasoning/thinking trace from a chat completion message.

    Tries each format in order, returning on the first match:

    * ``message.reasoning`` — vLLM / gpt-oss Harmony format.
    * ``message.reasoning_content`` — Ollama and other OpenAI-compatible servers.
    * ``<think>…</think>`` tags in ``message.content`` — Qwen3 and similar models.
    * Neither — models without reasoning; thinking is returned as ``""``.

    Returns ``(thinking, cleaned_content)`` where *cleaned_content* has any
    ``<think>`` block stripped so it is not shown to the user.
    """
    raw_content = msg.content or ""

    # 1. vLLM / gpt-oss Harmony format
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        return str(reasoning).strip(), raw_content

    # 2. Ollama-style dedicated field
    reasoning_content = getattr(msg, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content).strip(), raw_content

    # 3. <think>…</think> embedded in content
    match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        before = raw_content[: match.start()].strip()
        after = raw_content[match.end() :].strip()
        cleaned = (before + (" " if before and after else "") + after).strip()
        return thinking, cleaned

    # 4. No thinking trace
    return "", raw_content


SYSTEM_PROMPT = (
    "You are a helpful gardening assistant with expertise in plants, pests, and "
    "gardening techniques, drawing on Royal Horticultural Society (RHS) content. "
    "When answering, use the retrieved knowledge to ground your response. "
    "If the retrieved information does not cover the question, say so clearly."
)

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_gardening_knowledge",
        "description": (
            "Search the RHS gardening knowledge base for information relevant "
            "to the user's question. Call this whenever you need factual gardening "
            "information before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query to retrieve relevant gardening content.",
                }
            },
            "required": ["query"],
        },
    },
}


def index_documents(
    collection: chromadb.Collection,
    text_splitter: TokenTextSplitter,
    records: list[dict[str, Any]],
    source_type: str = "advice",
) -> None:
    """Index scraped JSONL records into the vector store.

    Parameters
    ----------
    collection:
        A ``chromadb.Collection`` instance (from ``get_retriever``).
    text_splitter:
        The text splitter used to chunk documents before indexing.
    records:
        List of scraped records. Schema depends on ``source_type``.
    source_type:
        One of ``"advice"``, ``"plants"``, or ``"pests"``.
    """
    documents = []
    for record in records:
        url = record.get("url", "")
        title = _get_title(record, source_type)
        body = _flatten_record(record, source_type)

        if not body.strip():
            continue

        page_content = f"# {title}\n\n{body}" if title else body
        documents.append(
            {
                "page_content": page_content,
                "metadata": {"source": url, "title": title},
            }
        )

    chunks = text_splitter.split_documents(documents)
    logger.info("Indexing %d chunks from %d documents...", len(chunks), len(documents))
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=[c["page_content"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[str(uuid.uuid4()) for _ in batch],
        )
    logger.info("Done indexing.")


class RAG:
    """Conversational RAG over RHS gardening content.

    Uses OpenAI tool calling to decide when to retrieve, maintaining per-thread
    conversation history as a plain list of OpenAI message dicts.

    Parameters
    ----------
    collection:
        A ``chromadb.Collection`` instance (from ``get_retriever``).
    client:
        An ``OpenAI`` client.
    model:
        Model name to use (default ``qwen3.5:4b``).
    k:
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        client: OpenAI,
        model: str = "qwen3.5:4b",
        k: int = 4,
    ):
        self.collection = collection
        self.client = client
        self.model = model
        self.k = k
        self._threads: dict[str, list[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str, thread_id: str = "default") -> ChatResult:
        """Send a message and return the assistant reply with any reasoning trace.

        Maintains conversation history per ``thread_id``. The LLM may call
        ``search_gardening_knowledge`` one or more times before returning a
        final answer.

        Reasoning traces are collected across all turns (including intermediate
        tool-call turns) and returned in ``ChatResult.thinking``. Works with
        vLLM (``reasoning`` field), Ollama (``reasoning_content`` field),
        embedded ``<think>`` tokens, and models with no reasoning (empty string).
        """
        history = self._threads[thread_id]

        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        history.append({"role": "user", "content": message})

        thinking_traces: list[str] = []

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=[_TOOL_SCHEMA],
                tool_choice="auto",
            )
            msg = response.choices[0].message
            turn_thinking, cleaned_content = _extract_thinking(msg)
            if turn_thinking:
                thinking_traces.append(turn_thinking)

            if msg.tool_calls:
                history.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self._search(args["query"])
                    logger.debug("Tool call %s → %d chars", tc.id, len(result))
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                history.append({"role": "assistant", "content": msg.content})
                thinking = "\n\n---\n\n".join(thinking_traces)
                return ChatResult(reply=cleaned_content, thinking=thinking)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str) -> str:
        """Retrieve docs and format them as a text block for the LLM."""
        results = self.collection.query(query_texts=[query], n_results=self.k)
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not docs:
            return "No relevant information found in the gardening knowledge base."

        snippets = []
        for text, meta in zip(docs, metadatas, strict=False):
            source = meta.get("source", "unknown")
            title = meta.get("title", "")
            header = f"[Source: {source}]" + (f" {title}" if title else "")
            snippets.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(snippets)
