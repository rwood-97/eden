"""Tests for generate_rag_distillation pure/mockable functions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from eden.synth_data_generation.generate_rag_distillation import (
    _cross_doc_prompt,
    _retrieve_context,
)

# ---------------------------------------------------------------------------
# _retrieve_context
# ---------------------------------------------------------------------------


def _make_collection(docs: list[str], metas: list[dict]) -> MagicMock:
    col = MagicMock()
    col.query.return_value = {"documents": [docs], "metadatas": [metas]}
    return col


def test_retrieve_context_returns_formatted_string():
    col = _make_collection(
        ["Roses need full sun."],
        [{"source": "https://rhs.org.uk/roses", "title": "Roses"}],
    )
    context, chunks = _retrieve_context(col, "rose care", k=1)

    assert "Roses need full sun." in context
    assert "rhs.org.uk/roses" in context
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Roses need full sun."


def test_retrieve_context_multiple_docs_separated():
    col = _make_collection(
        ["Doc one content.", "Doc two content."],
        [
            {"source": "https://rhs.org.uk/one", "title": "One"},
            {"source": "https://rhs.org.uk/two", "title": "Two"},
        ],
    )
    context, chunks = _retrieve_context(col, "query", k=2)

    assert "---" in context
    assert len(chunks) == 2


def test_retrieve_context_no_results_returns_fallback():
    col = _make_collection([], [])
    context, chunks = _retrieve_context(col, "unknown topic", k=4)

    assert "No relevant information" in context
    assert chunks == []


def test_retrieve_context_chunk_metadata_captured():
    col = _make_collection(
        ["Aphid info."],
        [{"source": "https://rhs.org.uk/pests/aphids", "title": "Aphids"}],
    )
    _, chunks = _retrieve_context(col, "aphids", k=1)

    assert chunks[0]["source"] == "https://rhs.org.uk/pests/aphids"
    assert chunks[0]["title"] == "Aphids"


# ---------------------------------------------------------------------------
# _cross_doc_prompt
# ---------------------------------------------------------------------------


CROSS_DOC_TEMPLATE = (
    "Below are {n_records} records.\n\n"
    "{record_blocks}\n\n"
    "Generate {n_questions} questions."
)


def test_cross_doc_prompt_includes_record_titles():
    records = [
        {"url": "https://rhs.org.uk/roses", "title": "Roses", "sections": []},
        {"url": "https://rhs.org.uk/pests", "title": "Aphids", "sections": []},
    ]
    prompt = _cross_doc_prompt(
        records, source_type="advice", n_questions=3, template=CROSS_DOC_TEMPLATE
    )

    assert "Roses" in prompt
    assert "Aphids" in prompt


def test_cross_doc_prompt_requests_correct_n_questions():
    records = [
        {"url": "https://rhs.org.uk/roses", "title": "Roses", "sections": []},
        {"url": "https://rhs.org.uk/pests", "title": "Aphids", "sections": []},
    ]
    prompt = _cross_doc_prompt(
        records, source_type="advice", n_questions=5, template=CROSS_DOC_TEMPLATE
    )

    assert "5" in prompt


def test_cross_doc_prompt_mentions_multiple_records():
    records = [
        {"url": "https://rhs.org.uk/a", "title": "A", "sections": []},
        {"url": "https://rhs.org.uk/b", "title": "B", "sections": []},
        {"url": "https://rhs.org.uk/c", "title": "C", "sections": []},
    ]
    prompt = _cross_doc_prompt(
        records, source_type="advice", n_questions=2, template=CROSS_DOC_TEMPLATE
    )

    assert "Record 1" in prompt
    assert "Record 2" in prompt
    assert "Record 3" in prompt


def test_cross_doc_prompt_substitutes_n_records():
    records = [
        {"url": "https://rhs.org.uk/a", "title": "A", "sections": []},
        {"url": "https://rhs.org.uk/b", "title": "B", "sections": []},
    ]
    prompt = _cross_doc_prompt(
        records, source_type="advice", n_questions=2, template=CROSS_DOC_TEMPLATE
    )

    assert "Below are 2 records." in prompt


# ---------------------------------------------------------------------------
# load_template
# ---------------------------------------------------------------------------


def test_load_template_fills_placeholders():
    from eden.synth_data_generation.generate_rag_distillation import load_template

    with tempfile.TemporaryDirectory() as tmp:
        tmpl = Path(tmp) / "synth_qa.txt"
        ideas = Path(tmp) / "question_ideas_advice.txt"
        tmpl.write_text(
            "Type: {source_type}. Count: {pairs_per_record}. {question_ideas_block}"
        )
        ideas.write_text("some question ideas")

        result = load_template(tmpl, source_type="advice", pairs_per_record=5)

    assert "advice" in result
    assert "5" in result
    assert "some question ideas" in result


def test_load_template_no_ideas_file():
    from eden.synth_data_generation.generate_rag_distillation import load_template

    with tempfile.TemporaryDirectory() as tmp:
        tmpl = Path(tmp) / "synth_qa.txt"
        # No question_ideas_advice.txt file created
        tmpl.write_text("Type: {source_type}. {question_ideas_block}End.")

        result = load_template(tmpl, source_type="advice", pairs_per_record=3)

    assert "advice" in result
    # The block should be replaced with empty string, leaving no stray placeholder
    assert "{question_ideas_block}" not in result
    assert "End." in result
