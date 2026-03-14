"""Sample ~1% of each dataset and report fields with content not captured by flatten_record."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eden.data_utils import flatten_record

SEED = 42
SAMPLE_RATE = 0.05

# Fields we know are metadata/numeric/structural — not useful text for the LLM
PLANTS_SKIP = {
    # Identifiers / internal
    "id",
    "plantEntityId",
    "synonymParentPlantId",
    "synonymParentPlantName",
    "isSynonym",
    "synonyms",
    "nameStatus",
    "scraped_at",
    # Search/autocomplete fields (redundant with content)
    "autoCompleteField",
    "autoCompleteFieldList",
    "semanticSearchField",
    # Numeric filter arrays (encode characteristics as integer codes, not text)
    "sunlight",
    "soilType",
    "spreadType",
    "heightType",
    "timeToFullHeight",
    "aspect",
    "moisture",
    "ph",
    "suggestedPlantUses",
    "plantingPlaces",
    "exposure",
    "plantType",
    "foliage",
    "habit",
    "seasonOfInterest",
    "seasonColourAgg",
    "colourWithAttributes",
    "hardinessLevel",
    # Booleans / nulls / pricing
    "price",
    "notedForFragrance",
    "nurseriesCount",
    "isAgm",
    "isGenus",
    "isSpecie",
    "isPlantsForPollinators",
    "isLowMaintenance",
    "isDroughtResistance",
    "hasFullProfile",
    "isNative",
    # Image metadata
    "images",
    "imageCopyRight",
    # Formatted HTML duplicate of botanicalNameUnFormatted
    "botanicalName",
    # Supplier link (not informational text)
    "supplierURL",
    # Empty in practice
    "hortGroupDescription",
    # Title fields — indexed as the document title, not in flatten_record body
    "botanicalNameUnFormatted",
    "commonName",
    "commonNameSortField",
    # genus is a short label redundant with genusDescription
    "genus",
}

ADVICE_SKIP = {
    "url",
    "title",
    "slug",
    "page_type",
    "scraped_at",
    # Cross-reference links — structure not content
    "related_problems",
}

PESTS_SKIP = {
    "url",
    "title",
    "slug",
    "type",
    "scraped_at",
    # Cross-reference links — structure not content
    "related_guides",
}


def has_content(value) -> bool:
    """Return True if the value contains meaningful text."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int | float)):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_content(v) for v in value)
    if isinstance(value, dict):
        return any(has_content(v) for v in value.values())
    return False


def text_from_flatten(record: dict, source_type: str) -> str:
    try:
        return flatten_record(record, source_type)
    except Exception:
        return ""


def check_dataset(path: Path, source_type: str, skip_fields: set[str]) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset: {source_type} ({path.name})")
    print(f"{'='*60}")

    records = []
    with open(path) as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                records.append(json.loads(line_stripped))

    random.seed(SEED)
    n = max(1, int(len(records) * SAMPLE_RATE))
    sample = random.sample(records, n)
    print(f"Total: {len(records):,}  |  Sample: {n:,} ({SAMPLE_RATE*100:.1f}%)")

    # Track which fields have content but aren't captured
    field_miss_count: Counter = (
        Counter()
    )  # field seen with content but not in flattened text
    field_total_count: Counter = Counter()  # field seen with content (denominator)
    novel_field_examples: dict[str, str] = {}

    for record in sample:
        flattened = text_from_flatten(record, source_type)
        # Strip HTML tags for a fair comparison
        flattened_plain = re.sub(r"<[^>]+>", "", flattened).lower()

        for field, value in record.items():
            if field in skip_fields:
                continue
            if not has_content(value):
                continue

            field_total_count[field] += 1

            # Extract one or more text probes from the value.
            # For dicts, probe each string value independently.
            probes: list[tuple[str, str]] = []  # (probe, raw_example)
            if isinstance(value, str):
                text = value.strip()
                plain = re.sub(r"<[^>]+>", "", text).lower().strip()
                if plain:
                    probes.append((plain[:40].strip(), text[:120]))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        plain = re.sub(r"<[^>]+>", "", item).lower().strip()
                        if plain:
                            probes.append((plain[:40].strip(), item[:120]))
                            break  # one representative probe per list field
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str) and v.strip():
                        plain = re.sub(r"<[^>]+>", "", v).lower().strip()
                        if plain:
                            probes.append((plain[:40].strip(), v[:120]))
                            break  # one representative probe per dict field

            if not probes:
                continue

            # A field is "missed" if none of its probes appear in the flattened text
            any_found = any(probe in flattened_plain for probe, _ in probes)
            if not any_found:
                field_miss_count[field] += 1
                if field not in novel_field_examples:
                    novel_field_examples[field] = probes[0][1]

    # Report fields missed in >20% of records where they had content
    missed = [
        (f, field_miss_count[f], field_total_count[f])
        for f in field_total_count
        if field_miss_count[f] / field_total_count[f] > 0.2
    ]
    missed.sort(key=lambda x: -x[1])

    if not missed:
        print("No significant gaps found — all content fields appear well covered.")
    else:
        print(
            "\nFields with content NOT captured by flatten_record (>20% miss rate):\n"
        )
        print(
            f"  {'Field':<30} {'Missed':>8} {'/ Total':>8}  {'Miss%':>6}  Example value"
        )
        print(f"  {'-'*30} {'-'*8} {'-'*8}  {'-'*6}  {'-'*40}")
        for field, missed_n, total_n in missed:
            pct = missed_n / total_n * 100
            example = novel_field_examples.get(field, "")
            example = example.replace("\n", " ")[:80]
            print(
                f"  {field:<30} {missed_n:>8} {('/ '+str(total_n)):>8}  {pct:>5.0f}%  {example}"
            )


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data" / "raw"
    check_dataset(base / "plants.jsonl", "plants", PLANTS_SKIP)
    check_dataset(base / "advice.jsonl", "advice", ADVICE_SKIP)
    check_dataset(base / "pests.jsonl", "pests", PESTS_SKIP)
