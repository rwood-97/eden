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

## Synthetic data generation

Synthetic QA pairs are generated from the scraped data using either an OpenAI-compatible API or Azure OpenAI.

### OpenAI-compatible backend (default)

Set the required environment variables (e.g. in a `.env` file):

```bash
OPENAI_API_BASE=<your-api-base-url>
OPENAI_API_KEY=<your-api-key>   # optional, defaults to "EMPTY"
```

### Azure OpenAI backend

```bash
AZURE_OPENAI_ENDPOINT=<your-azure-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-api-key>
```

Then run:

```bash
python -m eden.synth_data_generation.generate_synthetic_queries
```

Output is written to `data/synth/` as a JSONL file named `{source_type}_{model}_{n_records}rec_{pairs_per_record}pairs.jsonl`.

| Flag | Description |
|------|-------------|
| `--source-type` | `advice` (default), `plants`, or `pests` |
| `--n-records N` | Number of source records to sample (default: all) |
| `--pairs-per-record N` | QA pairs per record (default: 5) |
| `--model NAME` | Model name for the API (default: `gpt-oss-120b`) |
| `--backend` | `openai` (default) or `azure` |
| `--source-path PATH` | Source JSONL file (default: `data/raw/{source_type}.jsonl`) |
| `--save-path PATH` | Output directory (default: `data/synth/`) |
| `--overwrite` | Overwrite existing output file |
| `-v` | Verbose logging |

Example — generate QA pairs from 5 advice records using Azure OpenAI:

```bash
python -m eden.synth_data_generation.generate_synthetic_queries --n-records 5 --pairs-per-record 2 --source-type advice --backend azure --model gpt-4o -v
```

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
