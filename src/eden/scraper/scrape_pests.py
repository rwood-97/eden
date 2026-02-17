"""Scrape pest and disease guides from the RHS website.

Strategy:
1. Discover page URLs via the RHS internal advice search API
2. Fetch each page and parse the structured HTML content
3. Write results as JSONL

RHS pest/disease pages live under three URL prefixes:
  /biodiversity/{slug}  — pests, beneficial insects, wildlife
  /disease/{slug}       — plant diseases (fungal, viral, bacterial)
  /problems/{slug}      — physiological problems, disorders

All share a common template:
  Quick Facts, What is it?, Symptoms, Control/Management, Biology,
  Related Guides

The advice search API at POST /api/advice/Search returns all pages
with their URLs. We filter for the three pest/disease prefixes.

Usage:
    python -m eden.scraper.scrape_pests
    python -m eden.scraper.scrape_pests --limit 10
    python -m eden.scraper.scrape_pests --output data/raw/pests.jsonl
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

import typer
from bs4 import BeautifulSoup

from eden.scraper import BASE_URL
from eden.scraper.utils import (
    discover_urls_from_advice_api,
    make_client,
    scrape_loop,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/raw/pests.jsonl")

# URL prefixes that identify pest/disease pages
PEST_PREFIXES = ("/biodiversity/", "/disease/", "/problems/")

# Pattern to match pest/disease page URLs
PEST_URL_PATTERN = re.compile(r"^/(biodiversity|disease|problems)/[\w-]+$")


def parse_pest_page(html: str, url: str) -> dict | None:
    """Parse a pest/disease HTML page into structured data."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("h1")
    if not title_tag:
        logger.warning("No h1 found: %s", url)
        return None
    title = title_tag.get_text(strip=True)

    # Determine type and slug from URL path
    path = url.replace(BASE_URL, "").strip("/")
    parts = path.split("/")
    page_type = parts[0] if parts else "unknown"
    slug = parts[1] if len(parts) >= 2 else "unknown"

    # Extract quick facts from the Quick facts section
    # Facts are in div.fact__body elements, formatted as "Key - Value"
    quick_facts = {}
    for section_el in soup.find_all("section", class_="article-section"):
        heading_tag = section_el.find("h2")
        if heading_tag and "Quick facts" in heading_tag.get_text():
            for fact_body in section_el.find_all("div", class_="fact__body"):
                text = fact_body.get_text(strip=True)
                if " - " in text:
                    key, _, value = text.partition(" - ")
                    quick_facts[key.strip()] = value.strip()
            break

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

    # Extract related guide links
    related_guides = []
    seen_urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = f"{BASE_URL}{href}"
        href_clean = href.split("?")[0].split("#")[0].rstrip("/")
        rel_path = href_clean.replace(BASE_URL, "")
        if PEST_URL_PATTERN.match(rel_path):
            link_text = a.get_text(strip=True)
            if link_text and href_clean != url and href_clean not in seen_urls:
                seen_urls.add(href_clean)
                related_guides.append({"name": link_text, "url": href_clean})

    # Extract meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"] if meta_desc and meta_desc.get("content") else ""

    if not sections and not quick_facts:
        logger.warning("No content found: %s", url)
        return None

    return {
        "url": url,
        "title": title,
        "type": page_type,
        "slug": slug,
        "description": description,
        "quick_facts": quick_facts,
        "sections": sections,
        "related_guides": related_guides,
    }


def scrape_pests(
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
            urls = discover_urls_from_advice_api(
                client,
                url_filter=lambda path: any(path.startswith(p) for p in PEST_PREFIXES),
                label="pest/disease",
            )

    scrape_loop(
        urls=urls,
        output=output,
        parse_fn=parse_pest_page,
        checkpoint=checkpoint,
        limit=limit,
        label="pest/disease pages",
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
    """Scrape pest and disease guides from the RHS website."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape_pests(output, limit, checkpoint=not no_checkpoint, urls_file=urls_file)


if __name__ == "__main__":
    app()
