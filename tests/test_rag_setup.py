"""Tests for rag/build_index.py and rag/build_retriever.py.

Heavy objects (HuggingFaceEmbeddings, SentenceTransformersTokenTextSplitter,
Chroma) are mocked so no ML models or databases are loaded.

When LangChain is removed, update the patch targets in each test to match the
new imports — the assertions stay the same.
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
# setup_embedding_model
# ---------------------------------------------------------------------------


def test_setup_embedding_model_returns_huggingface_embeddings():
    with patch("eden.rag.build_index.HuggingFaceEmbeddings") as mock_hf:
        from eden.rag.build_index import setup_embedding_model

        setup_embedding_model("some-model")

        mock_hf.assert_called_once_with(model_name="some-model")


def test_setup_embedding_model_default_model():
    with patch("eden.rag.build_index.HuggingFaceEmbeddings") as mock_hf:
        from eden.rag.build_index import setup_embedding_model

        setup_embedding_model()

        call_kwargs = mock_hf.call_args.kwargs
        assert "all-mpnet-base-v2" in call_kwargs["model_name"]


# ---------------------------------------------------------------------------
# setup_text_splitter
# ---------------------------------------------------------------------------


def test_setup_text_splitter_returns_splitter():
    with patch(
        "eden.rag.build_index.SentenceTransformersTokenTextSplitter"
    ) as mock_splitter:
        from eden.rag.build_index import setup_text_splitter

        setup_text_splitter("my-model", chunk_overlap=25)

        mock_splitter.assert_called_once_with(model_name="my-model", chunk_overlap=25)


def test_setup_text_splitter_default_overlap():
    with patch(
        "eden.rag.build_index.SentenceTransformersTokenTextSplitter"
    ) as mock_splitter:
        from eden.rag.build_index import setup_text_splitter

        setup_text_splitter("my-model")

        call_kwargs = mock_splitter.call_args.kwargs
        assert call_kwargs["chunk_overlap"] == 50


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
    """Return a context manager stack patching all heavy deps in get_retriever."""
    return (
        patch("eden.rag.build_retriever.setup_embedding_model"),
        patch("eden.rag.build_retriever.setup_text_splitter"),
        patch("eden.rag.build_retriever.Chroma"),
    )


def test_get_retriever_returns_three_tuple():
    p_emb, p_spl, p_chroma = _patch_retriever_deps()
    with p_emb, p_spl, p_chroma:
        from eden.rag.build_retriever import DEFAULT_RETRIEVER_CONFIG, get_retriever

        result = get_retriever(DEFAULT_RETRIEVER_CONFIG)

        assert len(result) == 3


def test_get_retriever_uses_config_model_name():
    p_emb, p_spl, p_chroma = _patch_retriever_deps()
    with p_emb as mock_emb, p_spl, p_chroma:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="custom-model",
            chunk_overlap=30,
            persist_directory=None,
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        mock_emb.assert_called_once_with("custom-model")


def test_get_retriever_passes_persist_dir_to_chroma():
    p_emb, p_spl, p_chroma = _patch_retriever_deps()
    with p_emb, p_spl, p_chroma as mock_chroma:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="model",
            chunk_overlap=50,
            persist_directory=Path("/tmp/mydb"),
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["persist_directory"] == "/tmp/mydb"


def test_get_retriever_no_persist_dir_passes_none():
    p_emb, p_spl, p_chroma = _patch_retriever_deps()
    with p_emb, p_spl, p_chroma as mock_chroma:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="model",
            chunk_overlap=50,
            persist_directory=None,
            k=4,
            search_type="similarity",
        )
        get_retriever(cfg)

        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["persist_directory"] is None


def test_get_retriever_calls_as_retriever_with_k_and_search_type():
    p_emb, p_spl, p_chroma = _patch_retriever_deps()
    with p_emb, p_spl, p_chroma as mock_chroma:
        from eden.rag.build_retriever import RetrieverConfig, get_retriever

        cfg = RetrieverConfig(
            embedding_model_name="model",
            chunk_overlap=50,
            persist_directory=None,
            k=8,
            search_type="mmr",
        )
        get_retriever(cfg)

        mock_chroma.return_value.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 8},
        )
