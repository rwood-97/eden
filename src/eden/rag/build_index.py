import logging
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


@dataclass
class VectorStoreConfig:
    embedding_model_name: str
    chunk_overlap: int
    persist_directory: str | Path | None


class TokenTextSplitter:
    """Token-based text splitter using a HuggingFace tokenizer.

    Parameters
    ----------
    model_name:
        HuggingFace model name whose tokenizer is used for counting tokens.
    tokens_per_chunk:
        Maximum number of tokens per chunk.
    chunk_overlap:
        Number of tokens to overlap between consecutive chunks.
    """

    def __init__(
        self,
        model_name: str,
        tokens_per_chunk: int = 256,
        chunk_overlap: int = 50,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + self.tokens_per_chunk, len(token_ids))
            chunks.append(self._tokenizer.decode(token_ids[start:end]))
            if end == len(token_ids):
                break
            start = end - self.chunk_overlap
        return [c for c in chunks if c.strip()]

    def split_documents(self, documents: list[dict]) -> list[dict]:
        """Split a list of ``{"page_content": ..., "metadata": ...}`` dicts into chunks."""
        chunks = []
        for doc in documents:
            for text in self.split_text(doc["page_content"]):
                chunks.append({"page_content": text, "metadata": doc["metadata"]})
        return chunks


def setup_text_splitter(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    chunk_overlap: int = 50,
) -> TokenTextSplitter:
    logging.info("Setting up text splitter: %s (overlap=%d)", model_name, chunk_overlap)
    return TokenTextSplitter(model_name=model_name, chunk_overlap=chunk_overlap)
