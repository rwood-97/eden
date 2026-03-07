import json
import logging
import uuid
from collections import defaultdict
from typing import Any

import chromadb
from openai import OpenAI

from eden.data_utils import flatten_record as _flatten_record
from eden.data_utils import get_title as _get_title
from eden.rag.build_index import TokenTextSplitter

logger = logging.getLogger(__name__)

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
        Model name to use (default ``gpt-4o-mini``).
    k:
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        client: OpenAI,
        model: str = "gpt-4o-mini",
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

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Send a message and return the assistant reply.

        Maintains conversation history per ``thread_id``. The LLM may call
        ``search_gardening_knowledge`` one or more times before returning a
        final answer.
        """
        history = self._threads[thread_id]

        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        history.append({"role": "user", "content": message})

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=[_TOOL_SCHEMA],
                tool_choice="auto",
            )
            msg = response.choices[0].message

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
                answer = msg.content or ""
                history.append({"role": "assistant", "content": answer})
                return answer

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
