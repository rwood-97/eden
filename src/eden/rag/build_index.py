import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


@dataclass
class VectorStoreConfig:
    embedding_model_name: str
    chunk_overlap: int
    persist_directory: str | Path | None


def setup_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> HuggingFaceEmbeddings:
    log_msg = f"Setting up embedding model: {model_name}"
    logging.info(log_msg)
    return HuggingFaceEmbeddings(model_name=model_name)


def setup_text_splitter(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    chunk_overlap: int = 50,
) -> SentenceTransformersTokenTextSplitter:
    log_msg = f"Setting up text splitter with model: {model_name} and chunk overlap: {chunk_overlap}"
    logging.info(log_msg)
    return SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        chunk_overlap=chunk_overlap,
    )
