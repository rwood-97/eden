"""Tests for rag/build_index.py and rag/build_retriever.py.

Heavy objects (SentenceTransformerEmbeddingFunction, AutoTokenizer, chromadb
clients) are mocked so no ML models or databases are loaded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# VectorStoreConfig
# ---------------------------------------------------------------------------


def test_vector_store_config_fields():
    from eden.rag.build_index import VectorStoreConfig

    cfg = VectorStoreConfig(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        chunk_overlap=50,
        persist_directory=None,
    )
    assert cfg.embedding_model_name == "sentence-transformers/all-mpnet-base-v2"
    assert cfg.chunk_overlap == 50
    assert cfg.persist_directory is None


def test_vector_store_config_accepts_path():
    from eden.rag.build_index import VectorStoreConfig

    cfg = VectorStoreConfig(
        embedding_model_name="model",
        chunk_overlap=10,
        persist_directory=Path("/tmp/chroma"),
    )
    assert cfg.persist_directory == Path("/tmp/chroma")


# ---------------------------------------------------------------------------
# TokenTextSplitter
# ---------------------------------------------------------------------------


def test_token_text_splitter_init_loads_tokenizer():
    with patch("eden.rag.build_index.AutoTokenizer") as mock_tok:
        from eden.rag.build_index import TokenTextSplitter

        TokenTextSplitter("my-model", tokens_per_chunk=128, chunk_overlap=20)

        mock_tok.from_pretrained.assert_called_once_with("my-model")


def test_token_text_splitter_stores_config():
    with patch("eden.rag.build_index.AutoTokenizer"):
        from eden.rag.build_index import TokenTextSplitter

        splitter = TokenTextSplitter("model", tokens_per_chunk=128, chunk_overlap=20)

        assert splitter.tokens_per_chunk == 128
        assert splitter.chunk_overlap == 20


def test_token_text_splitter_split_documents_structure():
    with patch("eden.rag.build_index.AutoTokenizer") as mock_tok:
        mock_tokenizer = mock_tok.from_pretrained.return_value
        mock_tokenizer.encode.return_value = list(range(10))  # 10 tokens
        mock_tokenizer.decode.side_effect = lambda ids: f"chunk({ids})"

        from eden.rag.build_index import TokenTextSplitter

        splitter = TokenTextSplitter("model", tokens_per_chunk=10, chunk_overlap=0)
        docs = [{"page_content": "some text", "metadata": {"source": "url1"}}]
        chunks = splitter.split_documents(docs)

        assert isinstance(chunks, list)
        assert all("page_content" in c and "metadata" in c for c in chunks)
        assert all(c["metadata"] == {"source": "url1"} for c in chunks)


# ---------------------------------------------------------------------------
# setup_text_splitter
# ---------------------------------------------------------------------------


def test_setup_text_splitter_returns_token_text_splitter():
    with patch("eden.rag.build_index.AutoTokenizer"):
        from eden.rag.build_index import TokenTextSplitter, setup_text_splitter

        splitter = setup_text_splitter("my-model", chunk_overlap=25)

        assert isinstance(splitter, TokenTextSplitter)
        assert splitter.chunk_overlap == 25


def test_setup_text_splitter_default_overlap():
    with patch("eden.rag.build_index.AutoTokenizer"):
        from eden.rag.build_index import setup_text_splitter

        splitter = setup_text_splitter("my-model")

        assert splitter.chunk_overlap == 50


# ---------------------------------------------------------------------------
# RetrieverConfig
# ---------------------------------------------------------------------------


def test_retriever_config_inherits_vector_store_config():
    from eden.rag.build_retriever import RetrieverConfig

    cfg = RetrieverConfig(
        embedding_model_name="model",
        chunk_overlap=50,
        persist_directory=None,
        k=4,
        search_type="similarity",
    )
    assert cfg.k == 4
    assert cfg.search_type == "similarity"


def test_default_retriever_config_values():
    from eden.rag.build_retriever import DEFAULT_RETRIEVER_CONFIG

    assert DEFAULT_RETRIEVER_CONFIG.k == 4
    assert DEFAULT_RETRIEVER_CONFIG.search_type == "similarity"
    assert DEFAULT_RETRIEVER_CONFIG.persist_directory is None
    assert "all-mpnet-base-v2" in DEFAULT_RETRIEVER_CONFIG.embedding_model_name


# ---------------------------------------------------------------------------
# get_retriever
# ---------------------------------------------------------------------------


def _patch_retriever_deps():
    return (
        patch("eden.rag.build_retriever.SentenceTransformerEmbeddingFunction"),
        patch("eden.rag.build_retriever.setup_text_splitter"),
        patch("eden.rag.build_retriever.chromadb"),
    )


def test_get_retriever_returns_two_tuple():
    p_ef, p_spl, p_chroma = _patch_retriever_deps()
    with p_ef, p_spl, p_chroma:
        from eden.rag.build_retriever import DEFAULT_RETRIEVER_CONFIG, get_retriever

        result = get_retriever(DEFAULT_RETRIEVER_CONFIG)

        assert len(result) == 2


def test_get_retriever_uses_config_model_name():
    p_ef, p_spl, p_chroma = _patch_retriever_deps()
    with p_ef as mock_ef, p_spl, p_chroma:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="custom-model",
            chunk_overlap=30,
            persist_directory=None,
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        mock_ef.assert_called_once_with(model_name="custom-model")


def test_get_retriever_uses_persistent_client_when_persist_dir_set():
    p_ef, p_spl, p_chroma = _patch_retriever_deps()
    with p_ef, p_spl, p_chroma as mock_chromadb:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="model",
            chunk_overlap=50,
            persist_directory=Path("/tmp/mydb"),
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/mydb")


def test_get_retriever_uses_ephemeral_client_when_no_persist_dir():
    p_ef, p_spl, p_chroma = _patch_retriever_deps()
    with p_ef, p_spl, p_chroma as mock_chromadb:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="model",
            chunk_overlap=50,
            persist_directory=None,
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        mock_chromadb.EphemeralClient.assert_called_once()
        mock_chromadb.PersistentClient.assert_not_called()


def test_get_retriever_creates_named_collection():
    p_ef, p_spl, p_chroma = _patch_retriever_deps()
    with p_ef, p_spl, p_chroma as mock_chromadb:
        from eden.rag.build_retriever import DEFAULT_RETRIEVER_CONFIG, get_retriever

        get_retriever(DEFAULT_RETRIEVER_CONFIG)

        db = mock_chromadb.EphemeralClient.return_value
        db.get_or_create_collection.assert_called_once()
        assert (
            db.get_or_create_collection.call_args.kwargs["name"]
            == "gardening_knowledge"
        )
