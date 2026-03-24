FROM python:3.12-slim

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files and install dependencies from lockfile
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv sync --extra rag --extra server

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy Linux-built Chroma index (built via: podman run --rm -v $(pwd)/data:/app/data eden
#   python -m eden.rag.cli build-index --source-dir data/raw --persist-dir data/chroma_linux)
COPY data/chroma_linux/ data/chroma/

EXPOSE 8080

# AZURE_OPENAI_API_BASE and AZURE_OPENAI_API_KEY must be set at runtime,
# e.g. via Azure Container Apps secrets / environment variables.
# Override --model to match your Azure deployment name.
CMD ["python", "-m", "eden.rag.cli", "serve", \
     "--persist-dir", "data/chroma", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend", "openai", \
     "--model", "mistral-small-2503"]
