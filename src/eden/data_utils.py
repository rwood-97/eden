"""Shared utilities for flattening scraped records into plain text."""

from __future__ import annotations

import re

# Sections in semanticSearchField that are already captured from dedicated fields.
# Everything else (Characteristics, Colors and features, Suggested uses, Special features)
# is unique decoded data we extract below.
_SEM_SKIP = re.compile(
    r"(Botanical name|Preferred Common Name|Family|Genus information|Genus"
    r"|Description|Cultivation|Care information):",
    re.IGNORECASE,
)
_SEM_SECTION = re.compile(
    r"(Characteristics|Colors and features|Suggested uses|Special features)"
    r":(.*?)(?=Characteristics:|Colors and features:|Suggested uses:"
    r"|Special features:|Care information:|$)",
    re.DOTALL,
)


def get_title(record: dict, source_type: str) -> str:
    """Return a human-readable title for a scraped record."""
    if source_type == "plants":
        return (
            record.get("commonName", "")
            or record.get("botanicalNameUnFormatted", "")
            or "Unknown plant"
        )
    return record.get("title", "")


def get_page_type(record: dict, source_type: str) -> str:
    """Return a page-type label for a scraped record."""
    if source_type == "advice":
        return record.get("page_type", record.get("slug", "advice"))
    if source_type == "pests":
        return record.get("type", "biodiversity")
    return "plant-profile"


def flatten_record(record: dict, source_type: str) -> str:
    """Flatten a scraped record into plain text for embedding or prompting."""
    if source_type not in {"advice", "plants", "pests"}:
        msg = f"Unknown source_type {source_type!r}; expected one of: advice, plants, pests"
        raise ValueError(msg)

    if source_type == "plants":
        parts = []

        def _str(v) -> str:
            return (v or "").strip()

        # Descriptive summaries
        for field, label in (
            ("entityDescription", "Description"),
            ("genusDescription", "Genus"),
        ):
            value = _str(record.get(field))
            if value:
                parts.append(f"## {label}\n{value}")

        # Key characteristics as a block
        char_lines = []
        for field, label in (
            ("family", "Family"),
            ("height", "Height"),
            ("spread", "Spread"),
            ("fragrance", "Fragrance"),
            ("range", "Native range"),
        ):
            value = _str(record.get(field))
            if value:
                char_lines.append(f"{label}: {value}")
        # Alternative common names (list of strings)
        raw_common_names = record.get("commonNames") or []
        if isinstance(raw_common_names, str):
            raw_common_names = [raw_common_names]
        common_names = ", ".join(n for n in raw_common_names if n and n.strip())
        if common_names:
            char_lines.append(f"Also known as: {common_names}")
        if char_lines:
            parts.append("## Characteristics\n" + "\n".join(char_lines))

        # Toxicity / safety
        toxicity = record.get("toxicity") or []
        if toxicity:
            tox_text = "\n".join(str(t) for t in toxicity if t)
            if tox_text:
                parts.append(f"## Toxicity\n{tox_text}")

        # Care information
        for field in ("cultivation", "pruning", "propagation"):
            value = _str(record.get(field))
            if value:
                parts.append(f"## {field.capitalize()}\n{value}")

        # Pest and disease resistance
        for field, label in (
            ("pestResistance", "Pest resistance"),
            ("diseaseResistance", "Disease resistance"),
        ):
            value = _str(record.get(field))
            if value:
                parts.append(f"## {label}\n{value}")

        # Decoded characteristics from semanticSearchField:
        # habit, plant type, soil type, aspect, hardiness, moisture, pH,
        # sunlight, exposure, colours, suggested uses, special features.
        sem = _str(record.get("semanticSearchField"))
        if sem:
            for m in _SEM_SECTION.finditer(sem):
                label = m.group(1).strip()
                # Rename to avoid duplicate "## Characteristics" heading
                if label == "Characteristics":
                    label = "Growing conditions"
                content = m.group(2).strip().rstrip(".")
                if content:
                    parts.append(f"## {label}\n{content}")

        # Synonyms
        synonyms = record.get("synonyms") or []
        syn_names = [
            re.sub(r"<[^>]+>", "", s["name"]).strip()
            for s in synonyms
            if isinstance(s, dict) and s.get("name")
        ]
        if syn_names:
            parts.append("## Synonyms\n" + ", ".join(syn_names))

        return "\n\n".join(parts)

    if source_type == "pests":
        parts = []
        description = record.get("description", "").strip()
        if description:
            parts.append(description)
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

    # advice
    parts = []
    description = record.get("description", "")
    if description:
        parts.append(description)
    for section in record.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading and content:
            parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)
