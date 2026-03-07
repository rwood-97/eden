"""CLI for the Eden RAG pipeline.

Commands
--------
build-index
    Load scraped JSONL records into a persistent Chroma vector store.
    Accepts either a single file or a source directory (all three source types).

chat
    Interactive conversational chat grounded in the indexed knowledge base.

Usage examples::

    # Index all three source types in one go
    eden-rag build-index --source-dir data/raw --persist-dir data/chroma

    # Index a single file
    eden-rag build-index --source-file data/raw/advice.jsonl --persist-dir data/chroma

    # Start an interactive chat session
    eden-rag chat --persist-dir data/chroma
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Eden RAG pipeline CLI.")

DEFAULT_PERSIST_DIR = Path("data/chroma")
DEFAULT_SOURCE_DIR = Path("data/raw")

_KNOWN_SOURCE_TYPES = {"advice", "plants", "pests"}


# ---------------------------------------------------------------------------
# build-index
# ---------------------------------------------------------------------------


def _load_and_index(
    source_file: Path,
    source_type: str,
    collection: object,
    text_splitter: object,
    n_records: int | None,
) -> int:
    """Load records from *source_file* and index them. Returns record count."""
    from eden.rag.rag import index_documents

    with open(source_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if n_records is not None:
        records = records[:n_records]
    logging.info(
        "Indexing %d records from %s (source_type=%s)",
        len(records),
        source_file,
        source_type,
    )
    index_documents(collection, text_splitter, records, source_type=source_type)  # type: ignore[arg-type]
    return len(records)


@app.command("build-index")
def build_index(
    source_file: Annotated[
        Path | None,
        typer.Option("--source-file", help="Path to a single scraped JSONL file."),
    ] = None,
    source_type: Annotated[
        str,
        typer.Option(
            "--source-type", help="advice | plants | pests (only with --source-file)."
        ),
    ] = "advice",
    source_dir: Annotated[
        Path | None,
        typer.Option(
            "--source-dir",
            help=(
                "Directory containing advice.jsonl, plants.jsonl, pests.jsonl. "
                "Indexes all present files. Mutually exclusive with --source-file."
            ),
        ),
    ] = None,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Directory to persist the Chroma index."),
    ] = DEFAULT_PERSIST_DIR,
    n_records: Annotated[
        int | None,
        typer.Option(
            "--n-records", help="Limit to first N records per file (default: all)."
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging.")] = False,
) -> None:
    """Index scraped JSONL records into a persistent Chroma store.

    Pass either --source-dir (indexes all source types found) or
    --source-file + --source-type (indexes one file).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if source_file and source_dir:
        typer.echo("Provide either --source-file or --source-dir, not both.", err=True)
        raise typer.Exit(1)

    if not source_file and not source_dir:
        typer.echo("Provide --source-file or --source-dir.", err=True)
        raise typer.Exit(1)

    # Resolve which files to index
    if source_dir:
        if not source_dir.exists():
            typer.echo(f"Source directory not found: {source_dir}", err=True)
            raise typer.Exit(1)
        targets: list[tuple[Path, str]] = [
            (source_dir / f"{stype}.jsonl", stype)
            for stype in ("advice", "plants", "pests")
            if (source_dir / f"{stype}.jsonl").exists()
        ]
        if not targets:
            typer.echo(
                f"No advice/plants/pests JSONL files found in {source_dir}", err=True
            )
            raise typer.Exit(1)
    else:
        if source_type not in _KNOWN_SOURCE_TYPES:
            typer.echo(
                f"Invalid --source-type {source_type!r}. Must be: advice, plants, pests.",
                err=True,
            )
            raise typer.Exit(1)
        if not source_file.exists():
            typer.echo(f"Source file not found: {source_file}", err=True)
            raise typer.Exit(1)
        targets = [(source_file, source_type)]

    # Late import so the CLI is usable without rag deps when just --help
    from eden.rag.build_retriever import (
        DEFAULT_RETRIEVER_CONFIG,
        RetrieverConfig,
        get_retriever,
    )

    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=persist_dir,
        k=DEFAULT_RETRIEVER_CONFIG.k,
        search_type=DEFAULT_RETRIEVER_CONFIG.search_type,
    )

    collection, text_splitter = get_retriever(config)

    total = 0
    for fpath, stype in targets:
        total += _load_and_index(fpath, stype, collection, text_splitter, n_records)

    typer.echo(f"Done. Indexed {total} records into {persist_dir}")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command("chat")
def chat(
    persist_dir: Annotated[
        Path,
        typer.Option(
            "--persist-dir", help="Chroma persist directory built with build-index."
        ),
    ] = DEFAULT_PERSIST_DIR,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model name."),
    ] = "qwen3.5:4b",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of chunks to retrieve per query."),
    ] = 4,
    backend: Annotated[
        str, typer.Option("--backend", help="API backend: openai, azure, or ollama.")
    ] = "ollama",
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging.")] = False,
) -> None:
    """Start an interactive gardening chat session grounded in the indexed knowledge base."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not persist_dir.exists():
        typer.echo(
            f"Persist directory not found: {persist_dir}\n"
            "Run 'eden-rag build-index' first.",
            err=True,
        )
        raise typer.Exit(1)

    import os

    from eden.azure_client import make_azure_client
    from eden.openai_client import make_client
    from eden.rag.build_retriever import (
        DEFAULT_RETRIEVER_CONFIG,
        RetrieverConfig,
        get_retriever,
    )
    from eden.rag.rag import RAG

    if backend not in {"openai", "azure", "ollama"}:
        err_msg = (
            f"Invalid backend: {backend!r}. Must be one of: openai, azure, ollama."
        )
        raise ValueError(err_msg)

    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=persist_dir,
        k=k,
        search_type=DEFAULT_RETRIEVER_CONFIG.search_type,
    )

    collection, _ = get_retriever(config)
    if backend == "azure":
        client = make_azure_client(model=model)
    elif backend == "ollama":
        client = make_client(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
    else:
        client = make_client()
    rag = RAG(
        collection=collection,
        client=client,
        model=model,
        k=config.k,
    )

    typer.echo("Eden gardening assistant. Type 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            typer.echo("Goodbye.")
            break

        try:
            result = rag.chat(user_input)
            if result.thinking:
                typer.echo(f"\n[Thinking]\n{result.thinking}\n")
            typer.echo(f"\nEden: {result.reply}\n")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            sys.exit(1)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command("serve")
def serve(
    persist_dir: Annotated[
        Path,
        typer.Option(
            "--persist-dir", help="Chroma persist directory built with build-index."
        ),
    ] = DEFAULT_PERSIST_DIR,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model name."),
    ] = "qwen3.5:4b",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of chunks to retrieve per query."),
    ] = 4,
    backend: Annotated[
        str, typer.Option("--backend", help="API backend: openai, azure, or ollama.")
    ] = "ollama",
    host: Annotated[str, typer.Option("--host", help="Host to bind to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging.")] = False,
) -> None:
    """Start the Eden web chat server."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not persist_dir.exists():
        typer.echo(
            f"Persist directory not found: {persist_dir}\n"
            "Run 'eden-rag build-index' first.",
            err=True,
        )
        raise typer.Exit(1)

    import os

    import uvicorn

    from eden.azure_client import make_azure_client
    from eden.openai_client import make_client
    from eden.rag.build_retriever import (
        DEFAULT_RETRIEVER_CONFIG,
        RetrieverConfig,
        get_retriever,
    )
    from eden.rag.rag import RAG
    from eden.rag.server import app as fastapi_app
    from eden.rag.server import configure

    if backend not in {"openai", "azure", "ollama"}:
        err_msg = (
            f"Invalid backend: {backend!r}. Must be one of: openai, azure, ollama."
        )
        raise ValueError(err_msg)

    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=persist_dir,
        k=k,
        search_type=DEFAULT_RETRIEVER_CONFIG.search_type,
    )

    collection, _ = get_retriever(config)

    if backend == "azure":
        client = make_azure_client(model=model)
    elif backend == "ollama":
        client = make_client(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
    else:
        client = make_client()

    rag = RAG(collection=collection, client=client, model=model, k=config.k)
    configure(rag)

    typer.echo(f"Eden gardening assistant running at http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
