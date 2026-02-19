import logging
import os
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from eden.rag.build_index import (
    VectorStoreConfig,
    setup_embedding_model,
    setup_text_splitter,
)


@dataclass
class RetrieverConfig(VectorStoreConfig):
    k: int
    search_type: str  # "similarity" | "mmr"


DEFAULT_RETRIEVER_CONFIG = RetrieverConfig(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    chunk_overlap=50,
    persist_directory=None,
    k=4,
    search_type="similarity",
)


def get_retriever(
    config: RetrieverConfig = DEFAULT_RETRIEVER_CONFIG,
) -> tuple[Chroma, VectorStoreRetriever]:
    """Create (or load from disk) a Chroma vectorstore and return both the
    store and a retriever. The store is needed to add documents later.

    Returns
    -------
    (vectorstore, retriever)
    """
    embedding_model = setup_embedding_model(config.embedding_model_name)
    text_splitter = setup_text_splitter(
        config.embedding_model_name, config.chunk_overlap
    )

    persist_dir = str(config.persist_directory) if config.persist_directory else None

    if persist_dir and os.path.exists(persist_dir):
        logging.info("Loading Chroma from '%s'", persist_dir)
    else:
        logging.info("Creating new Chroma vectorstore")

    vectorstore = Chroma(
        collection_name="gardening_knowledge",
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

    retriever = vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs={"k": config.k},
    )

    return vectorstore, retriever, text_splitter
