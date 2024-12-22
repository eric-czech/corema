"""Shared test fixtures and utilities."""

import pytest
from unittest.mock import patch
from typing import Any, Generator


@pytest.fixture
def block_openai_api() -> Generator[None, None, None]:
    """Block any OpenAI API calls during testing.

    This fixture will raise a RuntimeError if any OpenAI API call is attempted.
    """

    def raise_on_openai_call(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("OpenAI API should not be called during testing!")

    with patch("openai.OpenAI.__call__", side_effect=raise_on_openai_call):
        yield
