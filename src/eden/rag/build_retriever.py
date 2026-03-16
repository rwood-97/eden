import logging
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from eden.rag.build_index import (
    TokenTextSplitter,
    VectorStoreConfig,
    setup_text_splitter,
)


@dataclass
class RetrieverConfig(VectorStoreConfig):
    k: int
    search_type: str  # "similarity" (kept for API compatibility)


DEFAULT_RETRIEVER_CONFIG = RetrieverConfig(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    chunk_overlap=50,
    persist_directory=None,
    k=4,
    search_type="similarity",
)


def get_retriever(
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
) -> tuple[chromadb.Collection, TokenTextSplitter]:
    """Create (or load from disk) a chromadb collection and text splitter.

    Returns
    -------
    (collection, text_splitter)
    """
    ef = SentenceTransformerEmbeddingFunction(model_name=config.embedding_model_name)
    text_splitter = setup_text_splitter(
        config.embedding_model_name, config.chunk_overlap
    )

    persist_dir = str(config.persist_directory) if config.persist_directory else None

    if persist_dir:
        logging.info("Loading/creating Chroma at '%s'", persist_dir)
        db = chromadb.PersistentClient(path=persist_dir)
    else:
        logging.info("Creating ephemeral chromadb")
        db = chromadb.EphemeralClient()

    collection = db.get_or_create_collection(
        name="gardening_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    return collection, text_splitter
