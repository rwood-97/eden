"""Shared utilities for Eden scrapers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from eden.scraper import (
    ADVICE_SEARCH_API,
    MAX_RETRIES,
    REQUEST_DELAY,
    RETRY_BASE_DELAY,
    USER_AGENT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def fetch_with_retries(client: httpx.Client, url: str) -> httpx.Response | None:
    """Fetch a URL with retries and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                logger.debug("404: %s", url)
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning("%s: %d, retrying in %ds", url, resp.status_code, delay)
                time.sleep(delay)
                continue
            logger.warning("%s: unexpected status %d", url, resp.status_code)
            return None
        except httpx.HTTPError as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning("%s: %s, retrying in %ds", url, e, delay)
            time.sleep(delay)

    logger.error("%s: failed after %d retries", url, MAX_RETRIES)
    return None


async def async_fetch_with_retries(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> httpx.Response | None:
    """Async fetch with retries, backoff, and concurrency limiting."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    logger.debug("404: %s", url)
                    return None
                if resp.status_code in (429, 500, 502, 503, 504):
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "%s: %d, retrying in %ds", url, resp.status_code, delay
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("%s: unexpected status %d", url, resp.status_code)
                return None
            except httpx.HTTPError as e:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning("%s: %s, retrying in %ds", url, e, delay)
                await asyncio.sleep(delay)

    logger.error("%s: failed after %d retries", url, MAX_RETRIES)
    return None


def make_client(**kwargs: Any) -> httpx.Client:
    """Create an httpx Client with default Eden headers."""
    kwargs.setdefault("headers", {})["User-Agent"] = USER_AGENT
    kwargs.setdefault("follow_redirects", True)
    kwargs.setdefault("timeout", 30)
    return httpx.Client(**kwargs)


def make_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with default Eden headers."""
    kwargs.setdefault("headers", {})["User-Agent"] = USER_AGENT
    kwargs.setdefault("follow_redirects", True)
    kwargs.setdefault("timeout", 30)
    return httpx.AsyncClient(**kwargs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-processed keys from checkpoint file."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(checkpoint_path: Path, key: str) -> None:
    """Append a key to the checkpoint file."""
    with open(checkpoint_path, "a") as f:
        f.write(f"{key}\n")


# ---------------------------------------------------------------------------
# Advice search API
# ---------------------------------------------------------------------------


def discover_urls_from_advice_api(
    client: httpx.Client,
    url_filter: Callable[[str], bool],
    label: str = "pages",
) -> list[str]:
    """Page through POST /api/advice/Search and collect matching URLs.

    Args:
        client: httpx Client to use.
        url_filter: Predicate applied to each hit's URL path.
        label: Label for log messages (e.g. "grow-your-own", "pest/disease").
    """
    from eden.scraper import BASE_URL

    all_urls: list[str] = []
    page_size = 200
    start = 0

    while True:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.post(
                    ADVICE_SEARCH_API,
                    json={
                        "profileIds": [],
                        "uniqueIds": [],
                        "startFrom": start,
                        "pageSize": page_size,
                    },
                )
                resp.raise_for_status()
                break
            except httpx.HTTPError as e:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning("API error: %s, retrying in %ds", e, delay)
                time.sleep(delay)
        else:
            logger.error("Failed to fetch advice search API after retries")
            break

        data = resp.json()
        total = data["totalHit"]
        hits = data["hits"]

        for hit in hits:
            url_path = hit.get("url", "")
            if url_filter(url_path):
                all_urls.append(f"{BASE_URL}{url_path}")

        start += len(hits)
        logger.info(
            "Discovery: %d/%d scanned, %d %s found",
            start,
            total,
            len(all_urls),
            label,
        )

        if start >= total or not hits:
            break

    logger.info("Discovered %d %s URLs", len(all_urls), label)
    return sorted(set(all_urls))


# ---------------------------------------------------------------------------
# Generic scrape loop
# ---------------------------------------------------------------------------


def scrape_loop(
    *,
    urls: list[str],
    output: Path,
    parse_fn: Callable[[str, str], dict | None],
    checkpoint: bool = True,
    limit: int | None = None,
    label: str = "pages",
) -> None:
    """Generic fetch-parse-write loop shared by advice and pest scrapers.

    Args:
        urls: Full list of URLs to scrape.
        output: JSONL output file path.
        parse_fn: Function(html, url) -> dict or None.
        checkpoint: Whether to use checkpoint/resume.
        limit: Max pages to scrape.
        label: Label for log messages.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".checkpoint")

    done = load_checkpoint(checkpoint_path) if checkpoint else set()
    remaining = [u for u in urls if u not in done]

    if done:
        logger.info(
            "Resuming: %d already done, %d remaining", len(done), len(remaining)
        )

    if limit is not None:
        remaining = remaining[:limit]

    logger.info("Scraping %d %s", len(remaining), label)

    scraped = 0
    skipped = 0
    start_time = time.monotonic()

    with make_client() as client:
        for i, url in enumerate(remaining):
            resp = fetch_with_retries(client, url)
            if resp is None:
                skipped += 1
                if checkpoint:
                    save_checkpoint(checkpoint_path, url)
                continue

            result = parse_fn(resp.text, url)
            if result is not None:
                result["scraped_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                with open(output, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                scraped += 1
            else:
                skipped += 1

            if checkpoint:
                save_checkpoint(checkpoint_path, url)

            elapsed = time.monotonic() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            if (i + 1) % 10 == 0 or i == len(remaining) - 1:
                logger.info(
                    "Progress: %d/%d (%.1f/s) — scraped: %d, skipped: %d",
                    i + 1,
                    len(remaining),
                    rate,
                    scraped,
                    skipped,
                )

            time.sleep(REQUEST_DELAY)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Done in %.1fs — scraped: %d, skipped: %d, output: %s",
        elapsed,
        scraped,
        skipped,
        output,
    )
