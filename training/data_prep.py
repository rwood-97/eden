"""Load and format SFT distillation data for training.

Converts the JSONL output of ``generate_rag_distillation.py`` into a
HuggingFace ``Dataset`` with fully-formatted chat strings ready for
``SFTTrainer``.

Each record's ``messages`` list is a complete tool-calling conversation
matching the production RAG inference format exactly:

    system  → RAG_SYSTEM_PROMPT
    user    → question
    assistant (tool call) → search_gardening_knowledge(query)
    tool    → retrieved context
    assistant → <think>…</think> response

Records with empty reasoning or very short responses are filtered out.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_MIN_RESPONSE_CHARS = 50
_MIN_REASONING_CHARS = 20


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_valid(record: dict) -> bool:
    """Return True if the record passes quality filters."""
    return (
        bool(record.get("question", "").strip())
        and bool(record.get("messages"))
        and len(record.get("response", "")) >= _MIN_RESPONSE_CHARS
        and len(record.get("reasoning", "")) >= _MIN_REASONING_CHARS
    )


def format_record(record: dict, tokenizer: PreTrainedTokenizerBase) -> str:
    """Apply the chat template to the saved tool-calling conversation."""
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(
    data_paths: list[Path],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 4096,
    eval_split: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load, filter, format, and split into train/eval datasets.

    Parameters
    ----------
    data_paths:
        One or more SFT JSONL files to combine.
    tokenizer:
        The model tokenizer used to apply the chat template.
    max_seq_length:
        Records exceeding this token count are dropped.
    eval_split:
        Fraction reserved for evaluation.
    seed:
        Random seed for the train/eval split.

    Returns
    -------
    (train_dataset, eval_dataset)
    """
    raw: list[dict] = []
    for path in data_paths:
        records = load_jsonl(path)
        logger.info("Loaded %d records from %s", len(records), path)
        raw.extend(records)

    before = len(raw)
    raw = [r for r in raw if is_valid(r)]
    logger.info("Kept %d/%d records after filtering", len(raw), before)

    texts: list[str] = []
    for record in raw:
        text = format_record(record, tokenizer)
        token_count = len(tokenizer.encode(text))
        if token_count > max_seq_length:
            logger.debug(
                "Dropping record (%d tokens > %d limit)", token_count, max_seq_length
            )
            continue
        texts.append(text)

    logger.info("Final dataset: %d examples", len(texts))

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    return split["train"], split["test"]
