"""Scrape grow-your-own advice articles from the RHS website.

Strategy:
1. Discover article URLs via the RHS internal advice search API
2. Fetch each page and parse the HTML content
3. Write results as JSONL

The RHS advice search API at POST /api/advice/Search returns all advice
articles with their URLs. We filter for grow-your-own guides which live at
paths like /vegetables/{slug}/grow-your-own, /fruit/{slug}/grow-your-own, etc.

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
    make_client,
    scrape_loop,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/raw/advice.jsonl")


def parse_advice_page(html: str, url: str) -> dict | None:
    """Parse an advice article HTML page into structured data."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("h1")
    if not title_tag:
        logger.warning("No h1 found: %s", url)
        return None
    title = title_tag.get_text(strip=True)

    # Determine category and slug from URL path
    path = url.replace(BASE_URL, "").strip("/")
    parts = path.split("/")
    category = parts[0] if parts else "unknown"
    slug = parts[1] if len(parts) >= 2 else "unknown"

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
        "category": category,
        "slug": slug,
        "description": description,
        "sections": sections,
        "related_problems": problem_links,
    }


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
            urls = discover_urls_from_advice_api(
                client,
                url_filter=lambda path: "/grow-your-own" in path,
                label="grow-your-own",
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
    """Scrape grow-your-own advice articles from the RHS website."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape_advice(output, limit, checkpoint=not no_checkpoint, urls_file=urls_file)


if __name__ == "__main__":
    app()
