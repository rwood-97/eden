"""Tests for training/data_prep.py.

Requires the ``training`` optional dependencies (datasets, transformers).
Skipped automatically when those are not installed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="training extras not installed")

from training.data_prep import build_dataset, is_valid, load_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> dict:
    base = {
        "question": "How do I grow tomatoes?",
        "messages": [
            {"role": "system", "content": "You are a gardening assistant."},
            {"role": "user", "content": "How do I grow tomatoes?"},
            {
                "role": "assistant",
                "content": "<think>reasoning</think>Plant in full sun.",
            },
        ],
        "reasoning": "I should look up tomato cultivation.",
        "response": "Plant tomatoes in full sun with well-drained soil.",
    }
    base.update(kwargs)
    return base


def _make_tokenizer(formatted_text: str = "<chat>formatted</chat>") -> MagicMock:
    tok = MagicMock()
    tok.apply_chat_template.return_value = formatted_text
    tok.encode.return_value = list(range(50))  # 50 tokens by default
    return tok


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_reads_all_records():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        _write_jsonl(path, [{"a": 1}, {"a": 2}])
        records = load_jsonl(path)
    assert len(records) == 2
    assert records[0] == {"a": 1}


def test_load_jsonl_skips_blank_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n')
        records = load_jsonl(path)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------


def test_is_valid_passes_good_record():
    assert is_valid(_make_record()) is True


def test_is_valid_rejects_empty_question():
    assert is_valid(_make_record(question="")) is False
    assert is_valid(_make_record(question="   ")) is False


def test_is_valid_rejects_missing_messages():
    assert is_valid(_make_record(messages=[])) is False
    assert is_valid(_make_record(messages=None)) is False


def test_is_valid_rejects_short_response():
    assert is_valid(_make_record(response="Too short.")) is False


def test_is_valid_rejects_short_reasoning():
    assert is_valid(_make_record(reasoning="brief")) is False


def test_is_valid_rejects_missing_reasoning():
    assert is_valid(_make_record(reasoning="")) is False


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_returns_train_eval_split():
    records = [_make_record() for _ in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        _write_jsonl(path, records)
        tok = _make_tokenizer()
        train, eval_ = build_dataset([path], tok, eval_split=0.2, seed=42)

    assert len(train) + len(eval_) == 20


def test_build_dataset_filters_invalid_records():
    records = [_make_record()] * 5 + [_make_record(response="bad")] * 3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        _write_jsonl(path, records)
        tok = _make_tokenizer()
        train, eval_ = build_dataset([path], tok, eval_split=0.2, seed=42)

    assert len(train) + len(eval_) == 5


def test_build_dataset_drops_long_sequences():
    records = [_make_record() for _ in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        _write_jsonl(path, records)
        # Tokenizer returns 200 tokens; set max_seq_length=100 to force all dropped
        tok = _make_tokenizer()
        tok.encode.return_value = list(range(200))
        train, eval_ = build_dataset([path], tok, max_seq_length=100, seed=42)

    assert len(train) == 0
    assert len(eval_) == 0


def test_build_dataset_combines_multiple_files():
    records = [_make_record() for _ in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.jsonl"
        p2 = Path(tmp) / "b.jsonl"
        _write_jsonl(p1, records[:5])
        _write_jsonl(p2, records[5:])
        tok = _make_tokenizer()
        train, eval_ = build_dataset([p1, p2], tok, eval_split=0.2, seed=42)

    assert len(train) + len(eval_) == 10


def test_build_dataset_calls_apply_chat_template():
    records = [_make_record() for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        _write_jsonl(path, records)
        tok = _make_tokenizer()
        build_dataset([path], tok, seed=42)

    assert tok.apply_chat_template.call_count == 5
    # Should pass the messages list, not tokenize
    call_kwargs = tok.apply_chat_template.call_args_list[0].kwargs
    assert call_kwargs.get("tokenize") is False
