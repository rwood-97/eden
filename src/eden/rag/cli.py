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
    rag: object,
    n_records: int | None,
) -> int:
    """Load records from *source_file* and index them. Returns record count."""
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
    rag.index_documents(records, source_type=source_type)  # type: ignore[attr-defined]
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
    backend: Annotated[
        str, typer.Option("--backend", help="API backend: openai or azure.")
    ] = "openai",
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
    import os

    from eden.azure_client import make_azure_client
    from eden.openai_client import make_client
    from eden.rag.build_retriever import (
        DEFAULT_RETRIEVER_CONFIG,
        RetrieverConfig,
        get_retriever,
    )
    from eden.rag.rag import RAG

    if backend not in {"openai", "azure"}:
        err_msg = f"Invalid backend: {backend!r}. Must be one of: openai, azure."
        raise ValueError(err_msg)

    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=persist_dir,
        k=DEFAULT_RETRIEVER_CONFIG.k,
        search_type=DEFAULT_RETRIEVER_CONFIG.search_type,
    )

    vectorstore, retriever, text_splitter = get_retriever(config)
    if backend == "azure":
        client = make_azure_client()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed-for-indexing")
        client = make_client(api_key=api_key)
    rag = RAG(
        vectorstore=vectorstore,
        retriever=retriever,
        text_splitter=text_splitter,
        client=client,
    )

    total = 0
    for fpath, stype in targets:
        total += _load_and_index(fpath, stype, rag, n_records)

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
    ] = "gpt-4o-mini",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of chunks to retrieve per query."),
    ] = 4,
    backend: Annotated[
        str, typer.Option("--backend", help="API backend: openai or azure.")
    ] = "openai",
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

    from eden.azure_client import make_azure_client
    from eden.openai_client import make_client
    from eden.rag.build_retriever import (
        DEFAULT_RETRIEVER_CONFIG,
        RetrieverConfig,
        get_retriever,
    )
    from eden.rag.rag import RAG

    if backend not in {"openai", "azure"}:
        err_msg = f"Invalid backend: {backend!r}. Must be one of: openai, azure."
        raise ValueError(err_msg)

    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=persist_dir,
        k=k,
        search_type=DEFAULT_RETRIEVER_CONFIG.search_type,
    )

    vectorstore, retriever, text_splitter = get_retriever(config)
    client = make_azure_client(model=model) if backend == "azure" else make_client()
    rag = RAG(
        vectorstore=vectorstore,
        retriever=retriever,
        text_splitter=text_splitter,
        client=client,
        model=model,
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
            reply = rag.chat(user_input)
            typer.echo(f"\nEden: {reply}\n")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    app()
