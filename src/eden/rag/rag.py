import json
import logging
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters.base import TextSplitter
from openai import OpenAI

load_dotenv()

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


# ---------------------------------------------------------------------------
# Record flattening helpers (mirrors synth_data_generation flatten logic)
# ---------------------------------------------------------------------------


def _get_title(record: dict, source_type: str) -> str:
    if source_type == "plants":
        return (
            record.get("commonName", "")
            or record.get("botanicalNameUnFormatted", "")
            or "Unknown plant"
        )
    return record.get("title", "")


def _flatten_record(record: dict, source_type: str) -> str:
    if source_type == "plants":
        parts = []
        for field in ("cultivation", "pruning", "propagation"):
            value = record.get(field, "").strip()
            if value:
                parts.append(f"## {field.capitalize()}\n{value}")
        return "\n\n".join(parts)

    if source_type == "pests":
        parts = []
        quick_facts = record.get("quick_facts", {})
        if quick_facts:
            facts_text = "\n".join(f"{k}: {v}" for k, v in quick_facts.items())
            parts.append(f"## Quick facts\n{facts_text}")
        for section in record.get("sections", []):
            heading = section.get("heading", "")
            content = section.get("content", "")
            if heading and content and heading.lower() != "quick facts":
                parts.append(f"## {heading}\n{content}")
        return "\n\n".join(parts)

    # advice (default)
    parts = []
    description = record.get("description", "")
    if description:
        parts.append(description)
    for section in record.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading:
            parts.append(f"## {heading}")
        if content:
            parts.append(content)
    return "\n\n".join(parts)


class RAG:
    """Conversational RAG over RHS gardening content.

    Uses OpenAI tool calling to decide when to retrieve, maintaining per-thread
    conversation history as a plain list of OpenAI message dicts.

    Parameters
    ----------
    vectorstore:
        A ``Chroma`` instance (from ``get_retriever``).
    retriever:
        A ``VectorStoreRetriever`` instance (from ``get_retriever``).
    text_splitter:
        The text splitter used to chunk documents before indexing.
    client:
        An ``OpenAI`` client.
    model:
        Model name to use (default ``gpt-4o-mini``).
    """

    def __init__(
        self,
        vectorstore: Chroma,
        retriever: VectorStoreRetriever,
        text_splitter: TextSplitter,
        client: OpenAI,
        model: str = "gpt-4o-mini",
    ):
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.text_splitter = text_splitter
        self.client = client
        self.model = model
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

    def index_documents(
        self,
        records: list[dict[str, Any]],
        source_type: str = "advice",
    ) -> None:
        """Index scraped JSONL records into the vector store.

        Parameters
        ----------
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
                Document(
                    page_content=page_content,
                    metadata={"source": url, "title": title},
                )
            )

        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            "Indexing %d chunks from %d documents...", len(chunks), len(documents)
        )
        self.vectorstore.add_documents(chunks)
        logger.info("Done indexing.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str) -> str:
        """Retrieve docs and format them as a text block for the LLM."""
        docs = self.retriever.invoke(query)
        if not docs:
            return "No relevant information found in the gardening knowledge base."

        snippets = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")
            header = f"[Source: {source}]" + (f" {title}" if title else "")
            snippets.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(snippets)
