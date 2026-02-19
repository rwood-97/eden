"""Generate synthetic QA pairs from RHS scraped data.

Usage:
    python -m eden.synth_data_generation.generate_synthetic_queries
    python -m eden.synth_data_generation.generate_synthetic_queries --n-records 5 --pairs-per-record 3 --source-type advice -v
"""

from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Annotated

import tqdm
import typer

from eden.synth_data_generation.azure_client import make_azure_client
from eden.synth_data_generation.openai_client import get_tool_response, make_client

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = Path("./templates/synth_qa.txt")
DEFAULT_SAVE_PATH = Path("./data/synth/")
DEFAULT_SOURCE_DATA_DIR = Path("./data/raw/")

_QA_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "save_qa_pairs",
        "description": "Save the generated question-answer pairs.",
        "parameters": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["pairs"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Content flattening
# ---------------------------------------------------------------------------


def _flatten_advice(record: dict) -> str:
    parts = []
    for section in record.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading and content:
            parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)


def _flatten_plants(record: dict) -> str:
    parts = []
    for field in ("cultivation", "pruning", "propagation"):
        value = record.get(field, "").strip()
        if value:
            parts.append(f"## {field.capitalize()}\n{value}")
    return "\n\n".join(parts)


def _flatten_pests(record: dict) -> str:
    parts = []
    quick_facts = record.get("quick_facts", {})
    if quick_facts:
        facts_text = "\n".join(f"{k}: {v}" for k, v in quick_facts.items())
        parts.append(f"## Quick facts\n{facts_text}")
    for section in record.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading and content and heading.lower() != "quick facts":
            parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)


_FLATTEN = {
    "advice": _flatten_advice,
    "plants": _flatten_plants,
    "pests": _flatten_pests,
}


def flatten_record(record: dict, source_type: str) -> str:
    fn = _FLATTEN.get(source_type)
    if fn is None:
        msg = f"Unknown source_type {source_type!r}; expected one of {list(_FLATTEN)}"
        raise ValueError(msg)
    return fn(record)


def get_title(record: dict, source_type: str) -> str:
    if source_type == "plants":
        common = record.get("commonName", "")
        botanical = record.get("botanicalNameUnFormatted", "")
        return common or botanical or "Unknown plant"
    return record.get("title", "")


def get_page_type(record: dict, source_type: str) -> str:
    if source_type == "advice":
        return record.get("page_type", record.get("slug", "advice"))
    if source_type == "pests":
        return record.get("type", "biodiversity")
    return "plant-profile"


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def load_template(template_path: Path, source_type: str, pairs_per_record: int) -> str:
    """Load the prompt template and fill in the static placeholders.

    Substitutes ``{question_ideas}``, ``{source_type}``, and
    ``{pairs_per_record}`` once upfront; ``{title}`` and ``{content}``
    are filled per record in the main loop.
    """
    template = template_path.read_text()
    ideas_path = template_path.parent / f"question_ideas_{source_type}.txt"
    question_ideas = ideas_path.read_text().strip()
    template = template.replace("{question_ideas}", question_ideas)
    template = template.replace("{source_type}", source_type)
    return template.replace("{pairs_per_record}", str(pairs_per_record))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_qa_pairs(
    n_records: int | None = None,
    pairs_per_record: int = 5,
    template_path: Path = DEFAULT_TEMPLATE,
    save_path: Path = DEFAULT_SAVE_PATH,
    source_path: Path = DEFAULT_SOURCE_DATA_DIR / "advice.jsonl",
    source_type: str = "advice",
    model: str = "gpt-oss-120b",
    backend: str = "openai",
    overwrite: bool = False,
) -> None:
    """Generate synthetic QA pairs from scraped RHS data.

    Parameters
    ----------
    n_records:
        Number of source records to sample. ``None`` uses all available records.
    pairs_per_record:
        QA pairs to generate per record.
    template_path:
        Path to the prompt template file.
    save_path:
        Directory for output JSONL files.
    source_path:
        Path to the source JSONL data file.
    source_type:
        One of ``"advice"``, ``"plants"``, or ``"pests"``.
    model:
        Model name for the API request.
    backend:
        One of ``"openai"`` (default) or ``"azure"``. Controls which client is used.
    overwrite:
        If ``False``, raise ``FileExistsError`` if the output file already exists.
    """
    template_path = Path(template_path)
    save_path = Path(save_path)
    source_path = Path(source_path)

    template = load_template(template_path, source_type, pairs_per_record)

    # Load and sample source records
    with open(source_path) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # For plants: filter to records with some text content
    if source_type == "plants":
        all_records = [
            r
            for r in all_records
            if any(
                r.get(k, "").strip() for k in ("cultivation", "pruning", "propagation")
            )
        ]

    if n_records is None:
        records = all_records
    else:
        sample_size = min(n_records, len(all_records))
        records = random.sample(all_records, sample_size)
    logger.info(
        "Using %d/%d records from %s", len(records), len(all_records), source_path
    )

    # Prepare output file
    save_path.mkdir(parents=True, exist_ok=True)
    n_label = "all" if n_records is None else n_records
    filename = f"{source_type}_{model}_{n_label}rec_{pairs_per_record}pairs.jsonl"
    out_path = save_path / filename

    if out_path.exists():
        if overwrite:
            logger.warning("Overwriting existing file: %s", out_path)
        else:
            err_msg = f"Output file already exists: {out_path}."
            raise FileExistsError(err_msg)

    client = make_azure_client() if backend == "azure" else make_client()

    with open(out_path, "w") as out:
        logger.info("Writing to %s", out_path)
        for record in tqdm.tqdm(records, desc="Generating QA pairs"):
            title = get_title(record, source_type)
            content = flatten_record(record, source_type)

            if not content.strip():
                logger.debug("Skipping record with no content: %s", title)
                continue

            prompt = template.replace("{title}", title).replace("{content}", content)

            raw = get_tool_response(client, prompt, model, _QA_TOOL)
            if raw is None:
                logger.warning("No response for: %s", title)
                continue

            try:
                pairs = json.loads(raw)["pairs"]
            except (json.JSONDecodeError, KeyError):
                logger.warning("Invalid tool response for: %s", title)
                continue

            page_type = get_page_type(record, source_type)
            generated_at = datetime.datetime.now(tz=datetime.UTC).isoformat()

            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                question = pair.get("question", "").strip()
                answer = pair.get("answer", "").strip()
                if not question or not answer:
                    continue
                out.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "source_url": record.get("url", ""),
                            "source_title": title,
                            "page_type": page_type,
                            "source_type": source_type,
                            "model": model,
                            "generated_at": generated_at,
                        }
                    )
                    + "\n"
                )

    logger.info("Done. Output: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    n_records: Annotated[
        int | None,
        typer.Option("--n-records", help="Source records to sample (default: all)"),
    ] = (None),
    pairs_per_record: Annotated[
        int, typer.Option("--pairs-per-record", help="QA pairs per record")
    ] = 5,
    template_path: Annotated[
        Path, typer.Option(help="Prompt template path")
    ] = DEFAULT_TEMPLATE,
    save_path: Annotated[
        Path, typer.Option(help="Output directory")
    ] = DEFAULT_SAVE_PATH,
    source_dir: Annotated[
        Path, typer.Option(help="Source JSONL file")
    ] = DEFAULT_SOURCE_DATA_DIR,
    source_type: Annotated[
        str, typer.Option(help="advice | plants | pests")
    ] = "advice",
    model: Annotated[str, typer.Option(help="Model name")] = "gpt-oss-120b",
    backend: Annotated[
        str, typer.Option(help="API backend: openai or azure")
    ] = "openai",
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing output")] = False,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging")] = False,
) -> None:
    """Generate synthetic QA pairs from RHS scraped data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if source_type not in {"advice", "plants", "pests"}:
        err_msg = f"Invalid source_type: {source_type!r}. Must be one of: advice, plants, pests."
        raise ValueError(err_msg)
    if backend not in {"openai", "azure"}:
        err_msg = f"Invalid backend: {backend!r}. Must be one of: openai, azure."
        raise ValueError(err_msg)

    generate_qa_pairs(
        n_records=n_records,
        pairs_per_record=pairs_per_record,
        template_path=template_path,
        save_path=save_path,
        source_path=source_dir / f"{source_type}.jsonl",
        source_type=source_type,
        model=model,
        backend=backend,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
