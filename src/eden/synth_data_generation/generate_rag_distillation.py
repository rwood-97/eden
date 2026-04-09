"""Generate SFT distillation data using RAG + large Qwen3.5 with reasoning.

Pipeline
--------
1. **Questions** — generated on-the-fly from raw RHS data.  Produces
   per-document questions (reusing the existing template) *and* cross-document
   questions that span multiple records (controlled by ``--cross-doc-fraction``).
2. **Retrieval** — each question is run through the Chroma index to fetch the
   top-k relevant chunks, exactly as the production RAG pipeline does.
3. **Answering** — the retrieved context is injected directly into the user
   message and sent to a large Qwen3.5 model with thinking enabled.  The
   ``<think>…</think>`` trace is extracted and saved alongside the response.
4. **Output** — one JSONL record per question containing the full prompt,
   context chunks, reasoning trace, and final response, ready for SFT.

Usage
-----
    python -m eden.synth_data_generation.generate_rag_distillation \\
        --source-type advice --chroma-dir data/chroma --source-dir data/raw
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Annotated, Any

import tqdm
import typer

from eden.azure_client import make_azure_client
from eden.data_utils import flatten_record, get_page_type, get_title
from eden.openai_client import get_tool_response, make_client
from eden.rag.build_retriever import (
    DEFAULT_RETRIEVER_CONFIG,
    RetrieverConfig,
    get_retriever,
)
from eden.rag.rag import SYSTEM_PROMPT, TOOL_SCHEMA, _extract_thinking

logger = logging.getLogger(__name__)

DEFAULT_SAVE_PATH = Path("./data/sft/")
DEFAULT_SOURCE_DATA_DIR = Path("./data/raw/")
DEFAULT_CHROMA_DIR = Path("./data/chroma/")
DEFAULT_TEMPLATE = Path("./templates/synth_qa.txt")

# ---------------------------------------------------------------------------
# Tool schemas for question generation
# ---------------------------------------------------------------------------

_QUESTION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "save_questions",
        "description": "Save the generated questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
    },
}


def load_template(template_path: Path, source_type: str, pairs_per_record: int) -> str:
    """Load the prompt template and fill in the static placeholders."""
    template = template_path.read_text()
    ideas_path = template_path.parent / f"question_ideas_{source_type}.txt"
    question_ideas = ideas_path.read_text().strip()
    template = template.replace("{question_ideas}", question_ideas)
    template = template.replace("{source_type}", source_type)
    return template.replace("{pairs_per_record}", str(pairs_per_record))


def _cross_doc_prompt(records: list[dict], source_type: str, n_questions: int) -> str:
    blocks = []
    for i, record in enumerate(records, 1):
        title = get_title(record, source_type)
        content = flatten_record(record, source_type)
        blocks.append(f"Record {i}: {title}\n{content[:800]}")
    combined = "\n\n".join(blocks)
    return (
        f"Below are {len(records)} RHS gardening records on related topics.\n\n"
        f"{combined}\n\n"
        f"Generate {n_questions} questions that require synthesising information "
        f"from more than one of the records above to answer fully. "
        f"Questions should be practical, specific, and useful to a gardener."
    )


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def _generate_per_doc_questions(
    records: list[dict],
    source_type: str,
    template: str,
    client: Any,
    model: str,
) -> list[dict]:
    """Generate per-document questions using tool calling (like the existing script)."""
    questions: list[dict] = []
    for record in tqdm.tqdm(records, desc="Generating per-doc questions"):
        title = get_title(record, source_type)
        content = flatten_record(record, source_type)
        if not content.strip():
            continue
        prompt = template.replace("{title}", title).replace("{content}", content)
        raw = get_tool_response(client, prompt, model, _QUESTION_TOOL)
        if raw is None:
            logger.warning("No response for: %s", title)
            continue
        try:
            qs = json.loads(raw)["questions"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Invalid tool response for: %s", title)
            continue
        page_type = get_page_type(record, source_type)
        for raw_q in qs:
            q = raw_q.strip() if isinstance(raw_q, str) else ""
            if q:
                questions.append(
                    {
                        "question": q,
                        "source_title": title,
                        "source_url": record.get("url", ""),
                        "page_type": page_type,
                        "source_type": source_type,
                        "question_type": "per_document",
                    }
                )
    return questions


def _generate_cross_doc_questions(
    records: list[dict],
    source_type: str,
    client: Any,
    model: str,
    n_questions: int,
    n_groups: int,
    collection: Any,
    url_to_record: dict[str, dict],
    group_size: int = 3,
) -> list[dict]:
    """Generate questions that span multiple semantically related records.

    For each group, a random seed record is chosen and its content is used to
    query the Chroma index.  The nearest-neighbour chunks (by embedding
    similarity) determine which records form the group — mirroring the
    documents that would co-occur in real RAG retrieval.
    """
    questions: list[dict] = []
    for _ in tqdm.trange(n_groups, desc="Generating cross-doc questions"):
        seed = random.choice(records)
        seed_text = flatten_record(seed, source_type)
        if not seed_text.strip():
            continue

        # Query Chroma with the seed content to find semantically related chunks.
        results = collection.query(
            query_texts=[seed_text[:1000]], n_results=group_size * 2
        )
        neighbour_urls = [m.get("source", "") for m in results["metadatas"][0]]

        # Map URLs back to full records; fall back to seed if no match.
        seen_urls: set[str] = set()
        group: list[dict] = []
        for url in neighbour_urls:
            if url in url_to_record and url not in seen_urls:
                group.append(url_to_record[url])
                seen_urls.add(url)
            if len(group) >= group_size:
                break

        if len(group) < 2:
            # Not enough distinct neighbours — fall back to seed + random pick.
            group = [seed, random.choice(records)]

        prompt = _cross_doc_prompt(group, source_type, n_questions)
        raw = get_tool_response(client, prompt, model, _QUESTION_TOOL)
        if raw is None:
            continue
        try:
            qs = json.loads(raw)["questions"]
        except (json.JSONDecodeError, KeyError):
            continue
        source_titles = [get_title(r, source_type) for r in group]
        source_urls = [r.get("url", "") for r in group]
        source_page_types = [get_page_type(r, source_type) for r in group]
        for q in qs:
            if isinstance(q, str) and q.strip():
                questions.append(
                    {
                        "question": q.strip(),
                        "source_titles": source_titles,
                        "source_urls": source_urls,
                        "page_types": source_page_types,
                        "source_type": source_type,
                        "question_type": "cross_document",
                    }
                )
    return questions


# ---------------------------------------------------------------------------
# RAG retrieval helper
# ---------------------------------------------------------------------------


def _retrieve_context(collection: Any, question: str, k: int) -> tuple[str, list[dict]]:
    """Query Chroma and return (formatted_context, chunk_metadata_list)."""
    results = collection.query(query_texts=[question], n_results=k)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not docs:
        return "No relevant information found in the gardening knowledge base.", []

    snippets = []
    chunks = []
    for text, meta in zip(docs, metadatas, strict=False):
        source = meta.get("source", "unknown")
        title = meta.get("title", "")
        header = f"[Source: {source}]" + (f" {title}" if title else "")
        snippets.append(f"{header}\n{text}")
        chunks.append({"source": source, "title": title, "text": text})

    return "\n\n---\n\n".join(snippets), chunks


# build_distillation_messages imported from eden.prompts.

# ---------------------------------------------------------------------------
# Async answering pipeline
# ---------------------------------------------------------------------------


async def _answer_question(
    question_data: dict,
    collection: Any,
    sync_client: Any,
    model: str,
    k: int,
    semaphore: asyncio.Semaphore,
    enable_thinking: bool,
) -> dict | None:
    """Run the agentic tool-calling loop with the large model, return SFT record.

    Mirrors rag.py's ``chat()`` loop so the saved ``messages`` list exactly
    matches the conversation format seen at inference time — eliminating any
    train/inference mismatch.
    """
    async with semaphore:
        question = question_data["question"]
        extra: dict = (
            {"extra_body": {"enable_thinking": True}} if enable_thinking else {}
        )

        history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        all_chunks: list[dict] = []

        # Agentic loop — model may call the search tool one or more times.
        while True:
            try:
                response = await asyncio.to_thread(
                    sync_client.chat.completions.create,
                    model=model,
                    messages=history,
                    tools=[TOOL_SCHEMA],
                    tool_choice="auto",
                    **extra,
                )
            except Exception:
                logger.exception("LLM call failed for: %s", question[:80])
                return None

            msg = response.choices[0].message

            if msg.tool_calls:
                # Model issued a search — execute and feed results back.
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
                    query = json.loads(tc.function.arguments)["query"]
                    context, chunks = await asyncio.to_thread(
                        _retrieve_context, collection, query, k
                    )
                    all_chunks.extend(chunks)
                    history.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": context}
                    )
            else:
                # Final answer — extract reasoning and append to history.
                reasoning, cleaned = _extract_thinking(msg)

                if not cleaned.strip():
                    logger.warning("Empty response for: %s", question[:80])
                    return None

                # Embed thinking trace in the assistant turn so the chat
                # template produces the correct <think>…</think> format for SFT.
                assistant_content = (
                    f"<think>\n{reasoning}\n</think>\n{cleaned}"
                    if reasoning
                    else cleaned
                )
                history.append({"role": "assistant", "content": assistant_content})

                record: dict = {
                    "question": question,
                    "messages": history,  # full tool-calling conversation for SFT
                    "context_chunks": all_chunks,
                    "reasoning": reasoning,
                    "response": cleaned,
                    "source_type": question_data.get("source_type", ""),
                    "question_type": question_data.get("question_type", ""),
                    "model": model,
                    "generated_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
                }
                if question_data.get("question_type") == "cross_document":
                    record["source_titles"] = question_data.get("source_titles", [])
                    record["source_urls"] = question_data.get("source_urls", [])
                    record["page_types"] = question_data.get("page_types", [])
                else:
                    record["source_title"] = question_data.get("source_title", "")
                    record["source_url"] = question_data.get("source_url", "")
                    record["page_type"] = question_data.get("page_type", "")
                return record


async def _run_answering(
    questions: list[dict],
    collection: Any,
    sync_client: Any,
    model: str,
    k: int,
    max_concurrent: int,
    enable_thinking: bool,
    out_path: Path,
) -> int:
    """Answer all questions concurrently and stream results to JSONL."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _answer_question(
            q, collection, sync_client, model, k, semaphore, enable_thinking
        )
        for q in questions
    ]

    written = 0
    with open(out_path, "w") as out:
        for coro in tqdm.tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Answering questions",
        ):
            record = await coro
            if record is not None:
                out.write(json.dumps(record) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_rag_distillation(
    source_type: str = "advice",
    source_dir: Path = DEFAULT_SOURCE_DATA_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    template_path: Path = DEFAULT_TEMPLATE,
    save_path: Path = DEFAULT_SAVE_PATH,
    model: str = "Qwen/Qwen3.5-122B-A10B-FP8",
    backend: str = "openai",
    k: int = 4,
    n_records: int | None = None,
    pairs_per_record: int = 5,
    cross_doc_fraction: float = 0.2,
    max_concurrent: int = 8,
    enable_thinking: bool = True,
    overwrite: bool = False,
    timeout: float = 600.0,
    questions_checkpoint: Path | None = None,
) -> None:
    """Generate SFT distillation data using RAG context and reasoning traces.

    Parameters
    ----------
    source_type:
        One of ``"advice"``, ``"plants"``, or ``"pests"``.
    source_dir:
        Directory containing raw JSONL files (used when generating questions).
    chroma_dir:
        Path to the persistent Chroma vector store.
    template_path:
        Prompt template for per-document question generation.
    save_path:
        Output directory.
    model:
        Large model to use for answering (e.g. ``Qwen/Qwen3.5-122B-A10B-FP8``).
    backend:
        API backend: ``"openai"``, ``"azure"``, or ``"ollama"``.
    k:
        Number of chunks to retrieve per question.
    n_records:
        Source records to sample for question generation (``None`` = all).
    pairs_per_record:
        Per-document questions to generate per source record.
    cross_doc_fraction:
        Fraction of total questions that should be cross-document.
    max_concurrent:
        Maximum concurrent LLM answering requests.
    enable_thinking:
        Pass ``enable_thinking=True`` in the request body (Qwen3 vLLM).
    overwrite:
        Overwrite existing output file.
    timeout:
        HTTP timeout in seconds for each LLM request. Increase for slow local models.
    """
    save_path = Path(save_path)
    chroma_dir = Path(chroma_dir)

    # Build client
    if backend == "azure":
        client = make_azure_client(model=model)
    elif backend == "ollama":
        client = make_client(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=timeout,
        )
    else:
        client = make_client(timeout=timeout)

    # Load Chroma retriever
    config = RetrieverConfig(
        embedding_model_name=DEFAULT_RETRIEVER_CONFIG.embedding_model_name,
        chunk_overlap=DEFAULT_RETRIEVER_CONFIG.chunk_overlap,
        persist_directory=chroma_dir,
        k=k,
        search_type="similarity",
    )
    collection, _ = get_retriever(config)

    # ---------------------------------------------------------------------------
    # Step 1: Generate questions (or load from checkpoint)
    # ---------------------------------------------------------------------------
    save_path.mkdir(parents=True, exist_ok=True)

    if questions_checkpoint is None:
        nr_tag = str(n_records) if n_records is not None else "all"
        cdf_tag = str(cross_doc_fraction).replace(".", "p")
        questions_checkpoint = (
            save_path
            / f"questions_{source_type}_{nr_tag}rec_{pairs_per_record}ppr_{cdf_tag}cdf.jsonl"
        )

    if questions_checkpoint.exists():
        logger.info("Loading questions from checkpoint: %s", questions_checkpoint)
        with open(questions_checkpoint) as f:
            questions = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d questions from checkpoint", len(questions))
    else:
        source_path = Path(source_dir) / f"{source_type}.jsonl"
        logger.info("Generating questions from %s", source_path)

        with open(source_path) as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        if source_type == "plants":
            all_records = [
                r
                for r in all_records
                if any(
                    r.get(k, "").strip()
                    for k in ("cultivation", "pruning", "propagation")
                )
            ]

        if n_records is not None:
            all_records = random.sample(all_records, min(n_records, len(all_records)))
        logger.info("Using %d source records", len(all_records))

        url_to_record = {r.get("url", ""): r for r in all_records if r.get("url")}

        template = load_template(Path(template_path), source_type, pairs_per_record)
        per_doc_qs = _generate_per_doc_questions(
            all_records, source_type, template, client, model
        )

        # Cross-document questions — groups formed by embedding similarity via Chroma.
        target_cross = int(
            len(per_doc_qs) * cross_doc_fraction / (1 - cross_doc_fraction)
        )
        n_groups = max(1, target_cross // 3)
        cross_doc_qs = _generate_cross_doc_questions(
            all_records,
            source_type,
            client,
            model,
            n_questions=3,
            n_groups=n_groups,
            collection=collection,
            url_to_record=url_to_record,
        )

        questions = per_doc_qs + cross_doc_qs
        random.shuffle(questions)
        logger.info(
            "Generated %d questions (%d per-doc, %d cross-doc)",
            len(questions),
            len(per_doc_qs),
            len(cross_doc_qs),
        )

        with open(questions_checkpoint, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        logger.info("Saved questions checkpoint: %s", questions_checkpoint)

    # ---------------------------------------------------------------------------
    # Step 2: Answer questions with RAG context + reasoning
    # ---------------------------------------------------------------------------
    save_path.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    filename = f"sft_{source_type}_{safe_model}_{len(questions)}q.jsonl"
    out_path = save_path / filename

    if out_path.exists():
        if overwrite:
            logger.warning("Overwriting existing file: %s", out_path)
        else:
            err_msg = (
                f"Output file already exists: {out_path}. Use --overwrite to replace."
            )
            raise FileExistsError(err_msg)

    logger.info("Writing SFT data to %s", out_path)
    written = asyncio.run(
        _run_answering(
            questions=questions,
            collection=collection,
            sync_client=client,
            model=model,
            k=k,
            max_concurrent=max_concurrent,
            enable_thinking=enable_thinking,
            out_path=out_path,
        )
    )
    logger.info("Done. Wrote %d/%d records to %s", written, len(questions), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    source_type: Annotated[
        str, typer.Option(help="advice, plants or pests")
    ] = "advice",
    source_dir: Annotated[
        Path,
        typer.Option(
            help="Path to Raw JSONL files for question generation (e.g. './data/raw/')"
        ),
    ] = DEFAULT_SOURCE_DATA_DIR,
    chroma_dir: Annotated[
        Path, typer.Option(help="Path to chroma persistent store directory")
    ] = DEFAULT_CHROMA_DIR,
    template_path: Annotated[
        Path, typer.Option(help="Path to question generation prompt template")
    ] = DEFAULT_TEMPLATE,
    save_path: Annotated[
        Path, typer.Option(help="Output directory")
    ] = DEFAULT_SAVE_PATH,
    model: Annotated[
        str,
        typer.Option(
            help="Model to use for answering (e.g. 'Qwen/Qwen3.5-122B-A10B-FP8')"
        ),
    ] = "Qwen/Qwen3.5-122B-A10B-FP8",
    backend: Annotated[
        str, typer.Option(help="API backend: openai, azure, or ollama")
    ] = "openai",
    k: Annotated[int, typer.Option(help="Chunks to retrieve per question")] = 4,
    n_records: Annotated[
        int | None,
        typer.Option(help="Source records to sample (default: all)"),
    ] = None,
    pairs_per_record: Annotated[
        int, typer.Option(help="Per-document questions per record")
    ] = 5,
    cross_doc_fraction: Annotated[
        float, typer.Option(help="Fraction of questions that are cross-document")
    ] = 0.4,
    max_concurrent: Annotated[
        int, typer.Option(help="Max concurrent LLM requests")
    ] = 8,
    enable_thinking: Annotated[
        bool, typer.Option(help="Pass enable_thinking=True to the model")
    ] = True,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing output file")
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            help="HTTP timeout in seconds per LLM request (increase for slow local models)"
        ),
    ] = 600.0,
    questions_checkpoint: Annotated[
        Path | None,
        typer.Option(
            help="Path to questions checkpoint JSONL. Default is auto-generated based on parameters."
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging")] = False,
) -> None:
    """Generate SFT distillation data with RAG context and reasoning traces."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if source_type not in {"advice", "plants", "pests"}:
        err_msg = f"Invalid source_type: {source_type!r}. Must be one of: advice, plants, pests."
        raise ValueError(err_msg)
    if backend not in {"openai", "azure", "ollama"}:
        err_msg = (
            f"Invalid backend: {backend!r}. Must be one of: openai, azure, ollama."
        )
        raise ValueError(err_msg)

    generate_rag_distillation(
        source_type=source_type,
        source_dir=source_dir,
        chroma_dir=chroma_dir,
        template_path=template_path,
        save_path=save_path,
        model=model,
        backend=backend,
        k=k,
        n_records=n_records,
        pairs_per_record=pairs_per_record,
        cross_doc_fraction=cross_doc_fraction,
        max_concurrent=max_concurrent,
        enable_thinking=enable_thinking,
        overwrite=overwrite,
        timeout=timeout,
        questions_checkpoint=questions_checkpoint,
    )


if __name__ == "__main__":
    app()
