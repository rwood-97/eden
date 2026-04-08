import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


def make_client(
    base_url: str | None = None, api_key: str | None = None, timeout: float = 600.0
) -> OpenAI:
    """Create an OpenAI-compatible client.

    Parameters
    ----------
    base_url:
        API base URL. Defaults to ``OPENAI_API_BASE`` env var, or
        ``http://localhost:8000/v1`` (vLLM default) if unset.
    api_key:
        API key. Defaults to ``OPENAI_API_KEY`` env var, or ``"EMPTY"`` for
        local servers that don't require auth.
    timeout:
        Request timeout in seconds. Increase for slow local models. Defaults to 600.
    """
    base_url = base_url or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def get_tool_response(
    client: OpenAI,
    prompt: str,
    model: str,
    tool: dict,
) -> str | None:
    """Send a prompt with a required tool call and return the arguments JSON string.

    Parameters
    ----------
    client:
        An ``OpenAI`` client instance.
    prompt:
        User prompt to send.
    model:
        Model name to use for the request.
    tool:
        OpenAI tool definition dict (``{"type": "function", "function": {...}}``).

    Returns
    -------
    str or None
        The raw ``arguments`` JSON string from the tool call, or ``None`` on failure.
    """
    tool_name = tool["function"]["name"]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            logger.debug(
                "Model did not return tool calls (model may not support function calling)"
            )
            return None
        return tool_calls[0].function.arguments
    except Exception:
        logger.exception("API call failed")
        return None
