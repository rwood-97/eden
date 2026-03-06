"""Tests for scraper HTML parse functions.

These are pure functions (HTML string → dict | None) so no network or mocking needed.
"""

from eden.scraper.scrape_advice import parse_advice_page
from eden.scraper.scrape_pests import parse_pest_page

# ---------------------------------------------------------------------------
# Fixtures — minimal valid HTML
# ---------------------------------------------------------------------------

ADVICE_HTML = """
<html>
<head>
  <meta name="description" content="A guide to growing roses.">
</head>
<body>
  <h1>Growing Roses</h1>
  <section class="article-section">
    <h2>Where to plant</h2>
    <div class="article-section__content">Choose a sunny spot.</div>
  </section>
  <section class="article-section">
    <h2>Watering</h2>
    <div class="article-section__content">Water at the base.</div>
  </section>
</body>
</html>
"""

PEST_HTML = """
<html>
<body>
  <h1>Aphids</h1>
  <section class="article-section">
    <h2>Quick facts</h2>
    <div class="fact__body">Common name - Greenfly</div>
    <div class="fact__body">Plants affected - Many garden plants</div>
  </section>
  <section class="article-section">
    <h2>Symptoms</h2>
    <div class="article-section__content">Sticky residue on leaves.</div>
  </section>
  <section class="article-section">
    <h2>Control</h2>
    <div class="article-section__content">Remove by hand or use insecticidal soap.</div>
  </section>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# parse_advice_page
# ---------------------------------------------------------------------------


def test_parse_advice_returns_dict():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    assert result is not None
    assert isinstance(result, dict)


def test_parse_advice_title():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    assert result["title"] == "Growing Roses"


def test_parse_advice_description():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    assert result["description"] == "A guide to growing roses."


def test_parse_advice_sections():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    headings = [s["heading"] for s in result["sections"]]
    assert "Where to plant" in headings
    assert "Watering" in headings


def test_parse_advice_section_content():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    section = next(s for s in result["sections"] if s["heading"] == "Where to plant")
    assert "Choose a sunny spot" in section["content"]


def test_parse_advice_page_type_classified():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    assert result["page_type"] == "grow-your-own"


def test_parse_advice_url_preserved():
    url = "https://www.rhs.org.uk/grow-your-own/roses"
    result = parse_advice_page(ADVICE_HTML, url)
    assert result["url"] == url


def test_parse_advice_no_h1_returns_none():
    html = "<html><body><section class='article-section'><h2>S</h2><div class='article-section__content'>X</div></section></body></html>"
    assert parse_advice_page(html, "https://www.rhs.org.uk/advice/test") is None


def test_parse_advice_no_sections_returns_none():
    html = "<html><body><h1>Title</h1></body></html>"
    assert parse_advice_page(html, "https://www.rhs.org.uk/advice/test") is None


def test_parse_advice_output_keys():
    result = parse_advice_page(
        ADVICE_HTML, "https://www.rhs.org.uk/grow-your-own/roses"
    )
    assert set(result.keys()) == {
        "url",
        "title",
        "page_type",
        "description",
        "sections",
        "related_problems",
    }


# ---------------------------------------------------------------------------
# parse_pest_page
# ---------------------------------------------------------------------------


def test_parse_pest_returns_dict():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    assert result is not None
    assert isinstance(result, dict)


def test_parse_pest_title():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    assert result["title"] == "Aphids"


def test_parse_pest_type_and_slug():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    assert result["type"] == "biodiversity"
    assert result["slug"] == "aphids"


def test_parse_pest_quick_facts():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    assert result["quick_facts"]["Common name"] == "Greenfly"
    assert result["quick_facts"]["Plants affected"] == "Many garden plants"


def test_parse_pest_sections():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    headings = [s["heading"] for s in result["sections"]]
    assert "Symptoms" in headings
    assert "Control" in headings


def test_parse_pest_output_keys():
    result = parse_pest_page(PEST_HTML, "https://www.rhs.org.uk/biodiversity/aphids")
    assert set(result.keys()) == {
        "url",
        "title",
        "type",
        "slug",
        "description",
        "quick_facts",
        "sections",
        "related_guides",
    }


def test_parse_pest_disease_url():
    html = """
    <html><body>
      <h1>Powdery Mildew</h1>
      <section class="article-section">
        <h2>Symptoms</h2>
        <div class="article-section__content">White powder on leaves.</div>
      </section>
    </body></html>
    """
    result = parse_pest_page(html, "https://www.rhs.org.uk/disease/powdery-mildew")
    assert result["type"] == "disease"
    assert result["slug"] == "powdery-mildew"


def test_parse_pest_no_content_returns_none():
    html = "<html><body><h1>Empty</h1></body></html>"
    assert parse_pest_page(html, "https://www.rhs.org.uk/biodiversity/empty") is None
