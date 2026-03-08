import json
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Iterator
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


class _ThinkStreamFilter:
    """Strips <think>…</think> blocks from a streaming content feed.

    Content outside think blocks is returned from ``push()`` for immediate
    emission; content inside think blocks is buffered and returned via
    ``get_reasoning()`` after ``flush()`` is called.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._reasoning: list[str] = []

    def push(self, text: str) -> str:
        """Feed a chunk; returns the portion that should be emitted as content."""
        self._buf += text
        output = ""
        while True:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    safe = max(0, len(self._buf) - len("</think>"))
                    self._reasoning.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                self._reasoning.append(self._buf[:end])
                self._buf = self._buf[end + len("</think>") :]
                self._in_think = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    safe = max(0, len(self._buf) - len("<think>"))
                    output += self._buf[:safe]
                    self._buf = self._buf[safe:]
                    break
                output += self._buf[:start]
                self._buf = self._buf[start + len("<think>") :]
                self._in_think = True
        return output

    def flush(self) -> str:
        """Flush at end of stream; returns any remaining emittable content."""
        if self._in_think:
            self._reasoning.append(self._buf)
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out

    def get_reasoning(self) -> str:
        return "".join(self._reasoning).strip()


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

    def chat_stream(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """Like ``chat()`` but yields SSE-style event dicts as the response streams.

        Yields dicts with a ``type`` key:

        * ``{"type": "searching", "query": "..."}`` — tool call in progress.
        * ``{"type": "token", "content": "..."}`` — streamed reply token.
        * ``{"type": "thinking", "content": "..."}`` — reasoning trace (after tokens).
        * ``{"type": "done"}`` — stream complete.
        """
        history = self._threads[thread_id]

        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        history.append({"role": "user", "content": message})

        thinking_traces: list[str] = []

        while True:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=[_TOOL_SCHEMA],
                tool_choice="auto",
                stream=True,
            )

            content_acc: list[str] = []
            reasoning_acc: list[str] = []
            tool_calls_map: dict[int, dict] = {}
            is_tool_turn = False
            think_filter = _ThinkStreamFilter()

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Dedicated reasoning fields (Ollama reasoning_content, vLLM reasoning)
                rc = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if rc:
                    reasoning_acc.append(rc)

                # Tool call deltas
                if delta.tool_calls:
                    is_tool_turn = True
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["arguments"] += (
                                    tc.function.arguments
                                )

                # Content tokens — only stream during final (non-tool) turns
                if delta.content and not is_tool_turn:
                    content_acc.append(delta.content)
                    to_emit = think_filter.push(delta.content)
                    if to_emit:
                        yield {"type": "token", "content": to_emit}

            # Flush any buffered content at end of stream
            if not is_tool_turn:
                remaining = think_filter.flush()
                if remaining:
                    yield {"type": "token", "content": remaining}

            full_content = "".join(content_acc)
            full_reasoning = "".join(reasoning_acc) or think_filter.get_reasoning()
            if full_reasoning:
                thinking_traces.append(full_reasoning.strip())

            if is_tool_turn:
                sorted_tcs = [tool_calls_map[i] for i in sorted(tool_calls_map)]
                history.append(
                    {
                        "role": "assistant",
                        "content": full_content or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in sorted_tcs
                        ],
                    }
                )
                for tc in sorted_tcs:
                    args = json.loads(tc["arguments"])
                    query = args["query"]
                    yield {"type": "searching", "query": query}
                    result = self._search(query)
                    logger.debug("Tool call %s → %d chars", tc["id"], len(result))
                    history.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
            else:
                clean = re.sub(
                    r"<think>.*?</think>", "", full_content, flags=re.DOTALL
                ).strip()
                history.append({"role": "assistant", "content": clean or full_content})
                if thinking_traces:
                    yield {
                        "type": "thinking",
                        "content": "\n\n---\n\n".join(thinking_traces),
                    }
                yield {"type": "done"}
                return

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
