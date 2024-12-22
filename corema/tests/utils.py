"""Test utilities."""

from openai import OpenAI


def check_openai_key(api_key: str | None) -> bool:
    """Check if OpenAI API key is valid.

    Args:
        api_key: The OpenAI API key to check

    Returns:
        bool: True if the key is valid, False otherwise.
    """
    try:
        if not api_key:
            return False
        client = OpenAI(api_key=api_key)
        client.models.list()  # Make a simple API call to verify the key
        return True
    except Exception:
        return False
