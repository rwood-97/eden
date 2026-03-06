import logging
import os

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"


def make_azure_client(
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str = DEFAULT_AZURE_API_VERSION,
    model: str | None = None,
) -> AzureOpenAI:
    """Create an Azure OpenAI client.

    ``base_url`` should be the deployments base URL, e.g.
    ``https://resource.openai.azure.com/openai/deployments/``.
    The ``model`` name is appended to form the full deployment URL.

    Parameters
    ----------
    base_url:
        Deployments base URL. Defaults to ``AZURE_OPENAI_API_BASE`` env var.
    api_key:
        Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY`` env var.
    api_version:
        Azure OpenAI API version. Defaults to ``DEFAULT_AZURE_API_VERSION``.
    model:
        Deployment/model name appended to ``base_url``.
    """
    base_url = base_url or os.environ["AZURE_OPENAI_API_BASE"]
    api_key = api_key or os.environ["AZURE_OPENAI_API_KEY"]
    if model:
        base_url = f"{base_url.rstrip('/')}/{model}/"
    return AzureOpenAI(
        base_url=base_url,
        api_key=api_key,
        api_version=api_version,
    )
