"""Shared utilities for flattening scraped records into plain text."""

from __future__ import annotations


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
        for field in ("cultivation", "pruning", "propagation"):
            value = record.get(field, "").strip()
            if value:
                parts.append(f"## {field.capitalize()}\n{value}")
        return "\n\n".join(parts)

    if source_type == "pests":
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
