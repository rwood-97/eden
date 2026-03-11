# eden

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A repo containing code related to fine-tuning and serving Eden (a LLM for gardening advice).

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

Optional dependency groups:

| Group | Contents | When to install |
|-------|----------|-----------------|
| `rag` | `chromadb`, `sentence-transformers` | RAG pipeline (build index, chat) |
| `server` | `fastapi`, `uvicorn` | Web chat UI |
| `dev` | `pytest`, `pre-commit` | Development |

Install specific groups:
```bash
uv sync --extra rag --extra server
```

## Scrapers

There are three scrapers to collect data from the RHS website into JSONL files under `data/raw/`.

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

## LLM set up

There are three options for which LLM backend to use for synthetic data generation and RAG:

1. Ollama (default) - local models via Ollama
2. OpenAI - e.g. when using vLLM
3. Azure OpenAI

To use these you will need to set the required environment variables in a `.env` file.

For `openai` backend:

```bash
OPENAI_API_BASE=<your-api-base-url>   # optional, defaults to http://localhost:8000/v1 (vLLM default)
OPENAI_API_KEY=<your-api-key>         # optional, defaults to "EMPTY"
```

For `azure` backend:

```bash
AZURE_OPENAI_ENDPOINT=<your-azure-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-api-key>
```

For `ollama` backend, no API key is required. Ensure Ollama is running locally (`ollama serve`). Optionally override the default URL:

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1   # optional, this is the default
```

## Fine-tuning

### Synthetic data generation

For fine-tuning, generate synthetic QA pairs using:

```bash
python -m eden.synth_data_generation.generate_synthetic_queries
```

Output is written to `data/synth/` as a JSONL file named `{source_type}_{model}_{n_records}rec_{pairs_per_record}pairs.jsonl`.

| Flag | Description |
|------|-------------|
| `--source-type` | `advice` (default), `plants`, or `pests` |
| `--n-records N` | Number of source records to sample (default: all) |
| `--pairs-per-record N` | QA pairs per record (default: 5) |
| `--model NAME` | Model name for the API (default: `qwen3.5:4b`) |
| `--backend` | `ollama` (default), `openai`, or `azure` |
| `--source-path PATH` | Source JSONL file (default: `data/raw/{source_type}.jsonl`) |
| `--save-path PATH` | Output directory (default: `data/synth/`) |
| `--overwrite` | Overwrite existing output file |
| `-v` | Verbose logging |

Example — generate QA pairs from plant records using OpenAI:

```bash
python -m eden.synth_data_generation.generate_synthetic_queries --source-type plants --pairs-per-record 2 --backend openai --model gpt-oss-120b -v
```

### Fine-tuning

Work in progress...

## RAG pipeline

### Build the index

Build the index for all three source types in one go using:

```bash
python -m eden.rag.cli build-index --source-dir data/raw --persist-dir data/chroma
```

Or, build the index per file (plants, advice, or pests) using:

```bash
python -m eden.rag.cli build-index --source-file data/raw/advice.jsonl --persist-dir data/chroma
```

| Flag | Description |
|------|-------------|
| `--source-dir PATH` | Directory containing `advice.jsonl`, `plants.jsonl`, `pests.jsonl` |
| `--source-file PATH` | Path to a single JSONL file |
| `--source-type` | `advice` (default), `plants`, or `pests` — only used with `--source-file` |
| `--persist-dir PATH` | Directory to write the Chroma index (default: `data/chroma`) |
| `--n-records N` | Limit to first N records per file (useful for testing) |
| `-v` | Verbose logging |

### Chat

Then run RAG chat using:

```bash
python -m eden.rag.cli chat --persist-dir data/chroma
```

| Flag | Description |
|------|-------------|
| `--persist-dir PATH` | Chroma index directory (default: `data/chroma`) |
| `--model NAME` | Model name (default: `qwen3.5:4b`) |
| `--k N` | Number of chunks to retrieve per query (default: 4) |
| `--backend` | `ollama` (default), `openai`, or `azure` |
| `-v` | Verbose logging |

Example — chat using a local OpenAI model:

```bash
python -m eden.rag.cli chat --persist-dir data/chroma --model gpt-oss-20b --backend openai
```

> **Note:** Ollama tool-calling support varies by model. Use models that support it (e.g. `qwen3.5:4b`, `llama3.1`, `llama3.2`, `mistral-nemo`).

### Web chat

Start a local web server with a browser-based chat UI (requires the `server` extras):

```bash
uv sync --extra rag --extra server
python -m eden.rag.cli serve --persist-dir data/chroma
```

Then open `http://localhost:8080` in your browser.

The UI supports multiple chat threads — use the sidebar to create new threads, switch between them, or delete old ones. Thread history is stored in the browser's `localStorage`.

| Flag | Description |
|------|-------------|
| `--persist-dir PATH` | Chroma index directory (default: `data/chroma`) |
| `--model NAME` | Model name (default: `qwen3.5:4b`) |
| `--k N` | Number of chunks to retrieve per query (default: 4) |
| `--backend` | `ollama` (default), `openai`, or `azure` |
| `--host HOST` | Host to bind to (default: `127.0.0.1`) |
| `--port PORT` | Port to listen on (default: `8080`) |
| `-v` | Verbose logging |

Example — web chat using a local Ollama model:

```bash
python -m eden.rag.cli serve --persist-dir data/chroma
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
