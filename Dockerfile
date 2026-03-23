FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files and install dependencies
COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system ".[rag,server]"

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy pre-built Chroma index.
# Build it locally first with:
#   eden-rag build-index --source-dir data/raw --persist-dir data/chroma
COPY data/chroma/ data/chroma/

EXPOSE 8080

# AZURE_OPENAI_API_BASE and AZURE_OPENAI_API_KEY must be set at runtime,
# e.g. via Azure Container Apps secrets / environment variables.
# Override --model to match your Azure deployment name.
CMD ["python", "-m", "eden.rag.cli", "serve", \
     "--persist-dir", "data/chroma", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--backend", "azure", \
     "--model", "gpt-5-nano"]
