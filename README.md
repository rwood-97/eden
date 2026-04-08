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

Then install the package with all extras:
```
uv sync --all-extras
```

This will install all optional dependency groups:

| Group | Contents | When to install |
|-------|----------|-----------------|
| `rag` | `chromadb`, `sentence-transformers` | RAG pipeline (build index, chat) |
| `server` | `fastapi`, `uvicorn` | Web chat UI |
| `training` | `torch`, `transformers`, `trl`, `accelerate`, `datasets` | SFT fine-tuning |
| `dev` | `pytest`, `pre-commit` | Development |

If you prefer you can install specific groups using:
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
AZURE_OPENAI_API_BASE=<your-azure-deployments-base-url>  # e.g. https://<resource>.cognitiveservices.azure.com/openai/deployments
AZURE_OPENAI_API_KEY=<your-azure-api-key>
```

For `ollama` backend, no API key is required. Ensure Ollama is running locally (`ollama serve`). Optionally override the default URL:

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1   # optional, this is the default
```

## Fine-tuning

Fine-tuning uses knowledge distillation (i.e. A large model generates questions, uses the RAG pipeline to generate conversations which are then used to fine-tune a smaller model).

The default large model is Qwen3.5-122B-A10B-FP8. Uses the official FP8 checkpoint, which fits on 4x H100 80GB GPUs (~122GB) with ~198GB free for KV cache and batching.

### Synthetic data generation

Install training dependencies from the `training` extra group:

```bash
uv sync --extra training
```

Then generate synthetic data from raw RHS content:

```bash
python -m eden.synth_data_generation.generate_rag_distillation \
    --source-type advice \
    --source-dir data/raw \
    --chroma-dir data/chroma \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --save-path data/sft/
```

You will need to repeat this for `--source-type plants` and `--source-type pests`.
By default, output is written to `data/sft/sft_{source_type}_{model}.jsonl`.

Key options:

| Flag | Description | Default |
|------|-------------|---------|
| `--source-type` | `advice`, `plants`, or `pests` | `advice` |
| `--model` | Large model for answering | `Qwen/Qwen3.5-122B-A10B-FP8` |
| `--backend` | `openai`, `azure`, or `ollama` | `openai` |
| `--k` | Chunks retrieved per question | `4` |
| `--n-records` | Source records to sample (default: all) | all |
| `--pairs-per-record` | Questions generated per source record | `5` |
| `--cross-doc-fraction` | Fraction of cross-document questions | `0.4` |
| `--max-concurrent` | Concurrent LLM requests | `8` |
| `--enable-thinking` | Request reasoning traces from the model | `True` |

### Fine-tuning Qwen3.5-4B

Training uses TRL `SFTTrainer` with PyTorch FSDP via `accelerate`.

Before running anything, you should edit `training/sft_config.yaml` to match your desired setup. The defaults are set to train Qwen/Qwen3.5-4B-Instruct for 3 epochs.

Key config values in `training/sft_config.yaml`:

| Key | Description | Default |
|-----|-------------|---------|
| `model_name` | HuggingFace model ID | `Qwen/Qwen3.5-4B-Instruct` |
| `data_paths` | List of SFT JSONL files | — |
| `output_dir` | Where to save the fine-tuned model | `models/qwen3.5-4b-eden/` |
| `num_train_epochs` | Training epochs | `3` |
| `max_seq_length` | Max tokens per example | `4096` |
| `per_device_train_batch_size` | Batch size per GPU | `4` |
| `gradient_accumulation_steps` | Gradient accumulation | `4` |
| `learning_rate` | Peak learning rate | `1e-5` |
| `eval_split` | Fraction held out for evaluation | `0.10` |
| `report_to` | Logging backend (`wandb`, `tensorboard`, `none`) | `wandb` |

Then launch your SLURM training script, the train command should look something like this:

```bash
NUM_GPUS=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
accelerate launch \
    --config_file training/fsdp_config.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_GPUS \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    training/sft_train.py --config training/sft_config.yaml
```

For a single-GPU smoke-test (no SLURM):

```bash
accelerate launch --num_processes 1 training/sft_train.py \
    --config training/sft_config.yaml --max-steps 5
```

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

## Container deployment

### Prerequisites

The container requires a Linux-built Chroma index. Build it using:

```bash
podman run --rm -v $(pwd)/data:/app/data eden \
  python -m eden.rag.cli build-index --source-dir data/raw --persist-dir data/chroma_linux
```

This writes the index to `data/chroma_linux/`, which is copied into the image at build time.

### Build and push the image

```bash
podman build -t ghcr.io/<your-github-username>/eden:latest .
podman push ghcr.io/<your-github-username>/eden:latest
```

### Run the container

The following environment variables must be set at runtime:

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_API_BASE` | Yes | Azure deployments base URL, e.g. `https://<resource>.cognitiveservices.azure.com/openai/deployments` |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `EDEN_PASSWORD` | No | If set, all `/chat` and `/chat/stream` requests must include an `X-Password` header matching this value |

Example (local):

```bash
podman run -p 8080:80 \
  -e AZURE_OPENAI_API_BASE=https://<resource>.cognitiveservices.azure.com/openai/deployments \
  -e AZURE_OPENAI_API_KEY=<key> \
  -e EDEN_PASSWORD=<password> \
  ghcr.io/<your-github-username>/eden:latest
```

### Password protection

When `EDEN_PASSWORD` is set, the server requires an `X-Password` header on all chat requests. The `/auth` endpoint can be used to validate a password before making chat requests:

```bash
curl -X POST http://localhost:8080/auth -H "X-Password: <password>"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

## Disclaimer

This work has been developed using Claude code and is based upon the [t0-1](https://github.com/alan-turing-institute/t0-1) repo.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/rwood-97/eden/workflows/CI/badge.svg
[actions-link]:             https://github.com/rwood-97/eden/actions
[pypi-link]:                https://pypi.org/project/eden/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/eden
[pypi-version]:             https://img.shields.io/pypi/v/eden
<!-- prettier-ignore-end -->
