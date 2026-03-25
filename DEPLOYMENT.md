# Deploying Eden to Azure Container Apps

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az`)
- [Podman](https://podman.io/) (or Docker)
- A GitHub Personal Access Token with `write:packages` scope
- An Azure subscription with a resource group
- An Azure AI Foundry deployment (or other OpenAI-compatible endpoint)

The following environment variables must be set (e.g. in a `.env` file):

```bash
GITHUB_PERSONAL_ACCESS_TOKEN=...
AZURE_OPENAI_API_BASE=https://<resource>.cognitiveservices.azure.com/openai/deployments
AZURE_OPENAI_API_KEY=...
EDEN_PASSWORD=...
```

## 1. Build the Linux Chroma index

The container requires a Chroma index built on Linux. Build it using Podman:

```bash
podman run --rm -v $(pwd)/data:/app/data ghcr.io/<your-username>/eden:latest \
  python -m eden.rag.cli build-index --source-dir data/raw --persist-dir data/chroma_linux
```

This writes the index to `data/chroma_linux/`, which is copied into the image at build time.

## 2. Build and push the image

```bash
set -a && source .env && set +a

podman build --platform linux/amd64 -t ghcr.io/<your-username>/eden:latest .

echo "$GITHUB_PERSONAL_ACCESS_TOKEN" | podman login ghcr.io -u <your-username> --password-stdin
podman push ghcr.io/<your-username>/eden:latest
```

> **Note:** Build with `--platform linux/amd64` even on Apple Silicon — ACI and Container Apps run on x86_64.

## 3. Create the Azure Container Apps environment

This only needs to be done once:

```bash
az containerapp env create \
  --name eden-env \
  --resource-group <your-resource-group> \
  --location uksouth
```

## 4. Deploy the Container App

```bash
set -a && source .env && set +a

az containerapp create \
  --name eden \
  --resource-group <your-resource-group> \
  --environment eden-env \
  --image ghcr.io/<your-username>/eden:latest \
  --target-port 80 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --registry-server ghcr.io \
  --registry-username <your-username> \
  --registry-password "$GITHUB_PERSONAL_ACCESS_TOKEN" \
  --secrets "azure-key=$AZURE_OPENAI_API_KEY" "eden-pw=$EDEN_PASSWORD" \
  --env-vars \
    "AZURE_OPENAI_API_BASE=$AZURE_OPENAI_API_BASE" \
    "AZURE_OPENAI_API_KEY=secretref:azure-key" \
    "EDEN_PASSWORD=secretref:eden-pw"
```

The app will be available at the HTTPS URL printed on completion, e.g.:
`https://eden.wittywave-c73785fb.uksouth.azurecontainerapps.io`

## 5. Updating the deployment

After rebuilding and pushing a new image:

```bash
az containerapp update \
  --name eden \
  --resource-group <your-resource-group> \
  --image ghcr.io/<your-username>/eden:latest
```

## Notes

### Port 80, not 8080
Azure Container Apps (and ACI) block port 8080 at the network level. The container must serve on port 80. The Dockerfile and `serve` command use `--port 80`.

### Environment variables
The azure backend reads `AZURE_OPENAI_API_BASE` and `AZURE_OPENAI_API_KEY`. Use Container Apps secrets for sensitive values rather than plain environment variables.
