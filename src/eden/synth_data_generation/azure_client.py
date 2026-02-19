import logging
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def make_azure_client(
    endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str = "2025-01-01-preview",
) -> AzureOpenAI:
    """Create an Azure OpenAI client.

    Parameters
    ----------
    endpoint:
        Azure OpenAI endpoint. Defaults to ``AZURE_OPENAI_ENDPOINT`` env var.
    api_key:
        Azure OpenAI API key. Defaults to ``AZURE_OPENAI_API_KEY`` env var.
    api_version:
        Azure OpenAI API version.
    """
    endpoint = endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = api_key or os.environ["AZURE_OPENAI_API_KEY"]
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
