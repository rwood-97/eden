import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters.base import TextSplitter


@dataclass
class VectorStoreConfig:
    embedding_model_name: str
    chunk_overlap: int
    persist_directory: str | Path | None


DEFAULT_VECTOR_STORE_CONFIG = VectorStoreConfig(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    chunk_overlap=50,
    persist_directory=None,
)


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


class VectorStoreCreator:
    def __init__(
        self,
        embedding_model: Embeddings,
        text_splitter: TextSplitter | None,
    ):
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        self.db: VectorStore | None = None

    def create_chroma(
        self,
        collection_name: str = "full_documents",
        config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
    ):
        from langchain_chroma import Chroma

        persist_dir = (
            str(config.persist_directory) if config.persist_directory else None
        )
        if persist_dir:
            log_msg = f"Persisting Chroma to '{persist_dir}'"
            logging.info(log_msg)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_dir,
        )
        return self.db

    def load_chroma(
        self,
        collection_name: str = "full_documents",
        config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
    ):
        from langchain_chroma import Chroma

        if config.persist_directory is None:
            err_msg = "persist_directory must be set to load Chroma."
            raise ValueError(err_msg)
        log_msg = f"Loading Chroma from '{config.persist_directory}'"
        logging.info(log_msg)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(config.persist_directory),
        )
        return self.db
