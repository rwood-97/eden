# eden

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A repo containing code related to fine-tuning Eden (a LLM for gardening advice).

## Installation

Clone the repo:
```bash
git clone https://github.com/rwood-97/eden
cd eden
```

Create a virtual environment using uv:
```bash
uv venv --python=3.12
source .venv/bin/activate
```

Then install the package:
```
uv sync --all-extras
```

## Scrapers

Three scrapers collect data from the RHS website into JSONL files under `data/raw/`.

You can use `python -m xxx` or `uv run xxx` to run the scrapers, where `xxx` is one of `eden.scraper.scrape_plants`, `eden.scraper.scrape_advice`, or `eden.scraper.scrape_pests`.

```bash
# Plants — uses sitemaps + RHS JSON API (~306k plants)
python -m eden.scraper.scrape_plants

# Grow-your-own advice articles (~108 pages)
python -m eden.scraper.scrape_advice

# Pest/disease guides (~403 pages)
python -m eden.scraper.scrape_pests
```

Common options (all scrapers):

| Flag | Description |
|------|-------------|
| `--output PATH` | Output JSONL file (default: `data/raw/{type}.jsonl`) |
| `--limit N` | Scrape at most N pages (useful for testing) |
| `--no-checkpoint` | Disable checkpoint/resume |
| `-v` | Verbose logging |

Scraping is resumable by default — a `.checkpoint` file tracks progress. Re-run the same command to pick up where you left off.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Disclaimer

This work has been co-developed using Claude code and is based upon the [t0-1](https://github.com/alan-turing-institute/t0-1) repo.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/rwood-97/eden/workflows/CI/badge.svg
[actions-link]:             https://github.com/rwood-97/eden/actions
[pypi-link]:                https://pypi.org/project/eden/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/eden
[pypi-version]:             https://img.shields.io/pypi/v/eden
<!-- prettier-ignore-end -->
