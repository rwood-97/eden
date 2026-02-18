"""Scrape gardening advice articles from the RHS website.

Strategy:
1. Discover article URLs via the RHS advice search API (denylist filter)
2. Discover in-month and beginners-guide URLs from sitemap-general.xml
3. Fetch each page and parse the HTML content
4. Write results as JSONL with page_type classification

The advice API returns ~1,400+ articles. We exclude pages already covered by
the pests scraper and non-gardening content, capturing ~800+ useful pages:
grow-your-own, growing-guide, pruning-guide, garden-design, plants/for-places,
propagation, lawns, container-gardening, garden-jobs, and more.

Pages not indexed in the advice API (in-month, beginners-guide) are discovered
from sitemap-general.xml.

Usage:
    python -m eden.scraper.scrape_advice
    python -m eden.scraper.scrape_advice --limit 10
    python -m eden.scraper.scrape_advice --output data/raw/advice.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from bs4 import BeautifulSoup

from eden.scraper import BASE_URL
from eden.scraper.utils import (
    discover_urls_from_advice_api,
    discover_urls_from_sitemap,
    make_client,
    scrape_loop,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/raw/advice.jsonl")

SITEMAP_URL = f"{BASE_URL}/sitemap-general.xml"

# Prefixes to exclude from the advice API — these are covered by other
# scrapers or are not gardening advice content.
EXCLUDED_PREFIXES = (
    "/biodiversity/",
    "/disease/",
    "/problems/",
    "/weeds/",
    "/education-learning/",
)

# Prefixes to discover from the sitemap (not indexed in the advice API).
SITEMAP_PREFIXES = (
    "/advice/in-month/",
    "/advice/beginners-guide/",
)

# Page type classification rules — checked in order, first match wins.
_PAGE_TYPE_RULES: list[tuple[str, str]] = [
    ("/grow-your-own", "grow-your-own"),
    ("/growing-guide", "growing-guide"),
    ("/pruning-guide", "pruning-guide"),
    ("/garden-design", "garden-design"),
    ("/in-month/", "in-month"),
    ("/beginners-guide/", "beginners-guide"),
    ("/plants/for-places/", "plants-for-places"),
    ("/propagation/", "propagation"),
    ("/garden-jobs/", "garden-jobs"),
]


def classify_page_type(url: str) -> str:
    """Derive a page_type label from URL path patterns."""
    for pattern, page_type in _PAGE_TYPE_RULES:
        if pattern in url:
            return page_type
    return "other"


def parse_advice_page(html: str, url: str) -> dict | None:
    """Parse an advice article HTML page into structured data."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("h1")
    if not title_tag:
        logger.warning("No h1 found: %s", url)
        return None
    title = title_tag.get_text(strip=True)

    page_type = classify_page_type(url)

    # Extract sections from article-section elements
    sections = []
    for section_el in soup.find_all("section", class_="article-section"):
        heading_tag = section_el.find("h2")
        if not heading_tag:
            continue
        heading = heading_tag.get_text(strip=True)
        if not heading:
            continue

        content_div = section_el.find("div", class_="article-section__content")
        if not content_div:
            continue

        content = content_div.get_text(strip=True)
        if content:
            sections.append({"heading": heading, "content": content})

    # Extract problem-solving links (links to /biodiversity/, /disease/, /problems/)
    problem_links = []
    seen_problem_urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = f"{BASE_URL}{href}"
        href = href.split("?")[0].split("#")[0]
        if any(p in href for p in ["/biodiversity/", "/disease/", "/problems/"]):
            link_text = a.get_text(strip=True)
            if link_text and href not in seen_problem_urls:
                seen_problem_urls.add(href)
                problem_links.append({"name": link_text, "url": href})

    # Extract meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

    if not sections:
        logger.warning("No sections found: %s", url)
        return None

    return {
        "url": url,
        "title": title,
        "page_type": page_type,
        "description": description,
        "sections": sections,
        "related_problems": problem_links,
    }


def _advice_url_filter(url: str) -> bool:
    """Accept advice API URLs except those matching excluded prefixes."""
    return not any(prefix in url for prefix in EXCLUDED_PREFIXES)


def _sitemap_url_filter(url: str) -> bool:
    """Accept sitemap URLs matching in-month or beginners-guide prefixes."""
    return any(prefix in url for prefix in SITEMAP_PREFIXES)


def scrape_advice(
    output: Path,
    limit: int | None = None,
    checkpoint: bool = True,
    urls_file: Path | None = None,
) -> None:
    """Main scraping pipeline."""
    if urls_file and urls_file.exists():
        with open(urls_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d URLs from %s", len(urls), urls_file)
    else:
        with make_client() as client:
            api_urls = discover_urls_from_advice_api(
                client,
                url_filter=_advice_url_filter,
                label="advice",
            )
            sitemap_urls = discover_urls_from_sitemap(
                client,
                url=SITEMAP_URL,
                url_filter=_sitemap_url_filter,
                label="in-month/beginners-guide",
            )
        urls = sorted(set(api_urls + sitemap_urls))
        logger.info(
            "Total: %d unique URLs (%d from API, %d from sitemap)",
            len(urls),
            len(api_urls),
            len(sitemap_urls),
        )

    scrape_loop(
        urls=urls,
        output=output,
        parse_fn=parse_advice_page,
        checkpoint=checkpoint,
        limit=limit,
        label="advice pages",
    )


app = typer.Typer()


@app.command()
def main(
    output: Annotated[
        Path, typer.Option(help="Output JSONL file path")
    ] = DEFAULT_OUTPUT,
    limit: Annotated[
        int | None, typer.Option(help="Max pages to scrape (for testing)")
    ] = None,
    no_checkpoint: Annotated[
        bool, typer.Option(help="Disable checkpoint/resume")
    ] = False,
    urls_file: Annotated[
        Path | None,
        typer.Option(help="File with URLs to scrape (one per line), skips discovery"),
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging")] = False,
) -> None:
    """Scrape gardening advice articles from the RHS website.

    Discovers pages from the advice API and sitemap, covering grow-your-own,
    growing-guide, pruning-guide, garden-design, in-month, beginners-guide,
    and more. Each record includes a page_type field for filtering.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape_advice(output, limit, checkpoint=not no_checkpoint, urls_file=urls_file)


if __name__ == "__main__":
    app()
