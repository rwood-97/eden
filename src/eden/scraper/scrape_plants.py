"""Scrape plant data from the RHS website via their internal API.

Strategy:
1. Parse sitemaps to discover all plant IDs
2. Fetch plant details from the JSON API for each ID
3. Write results as JSONL

Usage:
    python -m eden.scraper.scrape_plants
    python -m eden.scraper.scrape_plants --limit 100
    python -m eden.scraper.scrape_plants --output data/raw/plants.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from bs4 import BeautifulSoup

from eden.scraper import (
    DETAIL_ENDPOINT,
    MAX_CONCURRENT,
    REQUEST_DELAY,
    USER_AGENT,
)
from eden.scraper.utils import (
    async_fetch_with_retries,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

SITEMAP_INDEX_URL = "https://www.rhs.org.uk/sitemap_index.xml"
DEFAULT_OUTPUT = Path("data/raw/plants.jsonl")


def get_plant_sitemap_urls(client: httpx.Client) -> list[str]:
    """Fetch the sitemap index and return plant sitemap URLs."""
    resp = client.get(SITEMAP_INDEX_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml-xml")
    return [loc.text for loc in soup.find_all("loc") if "sitemap-plants" in loc.text]


def extract_plant_ids_from_sitemap(client: httpx.Client, url: str) -> list[int]:
    """Parse a plant sitemap XML and extract plant IDs from URLs."""
    resp = client.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml-xml")
    ids = []
    for loc in soup.find_all("loc"):
        match = re.search(r"/plants/(\d+)/", loc.text)
        if match:
            ids.append(int(match.group(1)))
    return ids


def discover_plant_ids(client: httpx.Client) -> list[int]:
    """Discover all plant IDs from sitemaps."""
    sitemap_urls = get_plant_sitemap_urls(client)
    logger.info("Found %d plant sitemaps", len(sitemap_urls))

    all_ids = []
    for url in sitemap_urls:
        ids = extract_plant_ids_from_sitemap(client, url)
        logger.info("  %s: %d plant IDs", url.split("/")[-1], len(ids))
        all_ids.extend(ids)

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_ids = []
    for pid in all_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    logger.info("Total unique plant IDs: %d", len(unique_ids))
    return unique_ids


async def fetch_plant_detail(
    client: httpx.AsyncClient,
    plant_id: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Fetch plant detail from the API with rate limiting and retries."""
    url = f"{DETAIL_ENDPOINT}/{plant_id}"
    resp = await async_fetch_with_retries(client, url, semaphore)
    if resp is None:
        return None
    return resp.json()


async def scrape_plants(
    output: Path,
    limit: int | None = None,
    checkpoint: bool = True,
) -> None:
    """Main scraping pipeline."""
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".checkpoint")

    # Discover plant IDs from sitemaps
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30,
    ) as sync_client:
        plant_ids = discover_plant_ids(sync_client)

    # Filter already-scraped IDs
    done_ids = load_checkpoint(checkpoint_path) if checkpoint else set()
    remaining_ids = [pid for pid in plant_ids if str(pid) not in done_ids]

    if done_ids:
        logger.info(
            "Resuming: %d already done, %d remaining", len(done_ids), len(remaining_ids)
        )

    if limit is not None:
        remaining_ids = remaining_ids[:limit]

    logger.info("Scraping %d plants", len(remaining_ids))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    scraped = 0
    skipped = 0
    start_time = time.monotonic()

    async with httpx.AsyncClient(
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
        follow_redirects=True,
        timeout=30,
    ) as client:
        # Process in batches to avoid overwhelming memory
        batch_size = 50
        for batch_start in range(0, len(remaining_ids), batch_size):
            batch = remaining_ids[batch_start : batch_start + batch_size]

            tasks = [fetch_plant_detail(client, pid, semaphore) for pid in batch]
            results = await asyncio.gather(*tasks)

            with open(output, "a") as f:
                for plant_id, result in zip(batch, results, strict=False):
                    if result is not None:
                        result["scraped_at"] = time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        )
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        scraped += 1
                    else:
                        skipped += 1

                    if checkpoint:
                        save_checkpoint(checkpoint_path, str(plant_id))

            elapsed = time.monotonic() - start_time
            total_done = batch_start + len(batch)
            rate = total_done / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f/s) — scraped: %d, skipped: %d",
                total_done,
                len(remaining_ids),
                rate,
                scraped,
                skipped,
            )

            # Rate limiting between batches
            await asyncio.sleep(REQUEST_DELAY)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Done in %.1fs — scraped: %d, skipped: %d, output: %s",
        elapsed,
        scraped,
        skipped,
        output,
    )


app = typer.Typer()


@app.command()
def main(
    output: Annotated[
        Path, typer.Option(help="Output JSONL file path")
    ] = DEFAULT_OUTPUT,
    limit: Annotated[
        int | None, typer.Option(help="Max plants to scrape (for testing)")
    ] = None,
    no_checkpoint: Annotated[
        bool, typer.Option(help="Disable checkpoint/resume")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging")] = False,
) -> None:
    """Scrape plant data from the RHS API."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(scrape_plants(output, limit, checkpoint=not no_checkpoint))


if __name__ == "__main__":
    app()
