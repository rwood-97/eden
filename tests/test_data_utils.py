"""Tests for eden.data_utils — pure functions, no mocking needed."""

import pytest

from eden.data_utils import flatten_record, get_page_type, get_title

# ---------------------------------------------------------------------------
# get_title
# ---------------------------------------------------------------------------


def test_get_title_advice():
    assert get_title({"title": "Growing Tomatoes"}, "advice") == "Growing Tomatoes"


def test_get_title_pests():
    assert get_title({"title": "Aphids"}, "pests") == "Aphids"


def test_get_title_plants_common_name():
    record = {"commonName": "Rose", "botanicalNameUnFormatted": "Rosa canina"}
    assert get_title(record, "plants") == "Rose"


def test_get_title_plants_botanical_fallback():
    record = {"commonName": "", "botanicalNameUnFormatted": "Rosa canina"}
    assert get_title(record, "plants") == "Rosa canina"


def test_get_title_plants_unknown_fallback():
    assert get_title({}, "plants") == "Unknown plant"


# ---------------------------------------------------------------------------
# get_page_type
# ---------------------------------------------------------------------------


def test_get_page_type_advice_uses_page_type_field():
    assert get_page_type({"page_type": "grow-your-own"}, "advice") == "grow-your-own"


def test_get_page_type_advice_falls_back_to_slug():
    assert get_page_type({"slug": "pruning-guide"}, "advice") == "pruning-guide"


def test_get_page_type_advice_default():
    assert get_page_type({}, "advice") == "advice"


def test_get_page_type_pests():
    assert get_page_type({"type": "disease"}, "pests") == "disease"


def test_get_page_type_pests_default():
    assert get_page_type({}, "pests") == "biodiversity"


def test_get_page_type_plants():
    assert get_page_type({}, "plants") == "plant-profile"


# ---------------------------------------------------------------------------
# flatten_record — advice
# ---------------------------------------------------------------------------


def test_flatten_advice_sections():
    record = {
        "description": "A useful guide.",
        "sections": [
            {"heading": "How to grow", "content": "Plant in full sun."},
            {"heading": "Watering", "content": "Water regularly."},
        ],
    }
    result = flatten_record(record, "advice")
    assert "A useful guide." in result
    assert "## How to grow\nPlant in full sun." in result
    assert "## Watering\nWater regularly." in result


def test_flatten_advice_no_description():
    record = {"sections": [{"heading": "Pruning", "content": "Cut back in spring."}]}
    result = flatten_record(record, "advice")
    assert "## Pruning\nCut back in spring." in result


def test_flatten_advice_skips_incomplete_sections():
    record = {
        "sections": [
            {"heading": "Title only", "content": ""},
            {"heading": "", "content": "Content only"},
            {"heading": "Complete", "content": "Both present."},
        ]
    }
    result = flatten_record(record, "advice")
    assert "Title only" not in result
    assert "Content only" not in result
    assert "## Complete\nBoth present." in result


def test_flatten_advice_empty_record():
    assert flatten_record({}, "advice") == ""


# ---------------------------------------------------------------------------
# flatten_record — plants
# ---------------------------------------------------------------------------


def test_flatten_plants_all_fields():
    record = {
        "cultivation": "Plant in well-drained soil.",
        "pruning": "Prune in late winter.",
        "propagation": "Propagate by seed.",
    }
    result = flatten_record(record, "plants")
    assert "## Cultivation\nPlant in well-drained soil." in result
    assert "## Pruning\nPrune in late winter." in result
    assert "## Propagation\nPropagate by seed." in result


def test_flatten_plants_partial_fields():
    record = {"cultivation": "Sandy soil preferred.", "pruning": "", "propagation": ""}
    result = flatten_record(record, "plants")
    assert "## Cultivation\nSandy soil preferred." in result
    assert "Pruning" not in result


def test_flatten_plants_empty_record():
    assert flatten_record({}, "plants") == ""


# ---------------------------------------------------------------------------
# flatten_record — pests
# ---------------------------------------------------------------------------


def test_flatten_pests_quick_facts_and_sections():
    record = {
        "quick_facts": {"Common name": "Greenfly", "Plants affected": "Roses"},
        "sections": [
            {"heading": "Symptoms", "content": "Sticky leaves and distortion."},
        ],
    }
    result = flatten_record(record, "pests")
    assert "## Quick facts" in result
    assert "Common name: Greenfly" in result
    assert "## Symptoms\nSticky leaves and distortion." in result


def test_flatten_pests_excludes_quick_facts_section_heading():
    """Section headings named 'Quick facts' should not be duplicated."""
    record = {
        "quick_facts": {"Host plants": "Many"},
        "sections": [
            {"heading": "Quick facts", "content": "Should be skipped."},
            {"heading": "Control", "content": "Use neem oil."},
        ],
    }
    result = flatten_record(record, "pests")
    assert result.count("Quick facts") == 1
    assert "## Control\nUse neem oil." in result


def test_flatten_pests_empty_record():
    assert flatten_record({}, "pests") == ""


# ---------------------------------------------------------------------------
# flatten_record — unknown source_type
# ---------------------------------------------------------------------------


def test_flatten_unknown_source_type_raises():
    with pytest.raises(ValueError, match="Unknown source_type"):
        flatten_record({}, "unknown")
