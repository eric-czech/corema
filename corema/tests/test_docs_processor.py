import pytest
from unittest.mock import Mock, create_autospec, patch
import requests
from typing import Any

from corema.processors.docs import DocsProcessor
from corema.storage import LocalStorage


@pytest.fixture
def storage() -> Mock:
    """Create a mock storage instance."""
    return create_autospec(LocalStorage)  # type: ignore[no-any-return]


@pytest.fixture
def processor(storage: Mock) -> DocsProcessor:
    """Create a DocsProcessor instance with mocked storage."""
    return DocsProcessor(storage)


def test_process_single_page(processor: DocsProcessor) -> None:
    """Test processing a single documentation page with no recursion."""
    with patch.object(processor.crawler, "get") as mock_get:
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <a href="/page2">Page 2</a>
                <a href="/page3">Page 3</a>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        processor.process_documentation(
            "test_project", "https://docs.example.com", max_depth=0
        )

        # Should only store the initial page
        processor.storage.store_text.assert_called_once_with(  # type: ignore[attr-defined]
            "test_project",
            mock_response.text,
            "https://docs.example.com",
            "docs",
            ".html",
        )


def test_process_one_level_deep(processor: DocsProcessor) -> None:
    """Test processing documentation pages one level deep."""
    with patch.object(processor.crawler, "get") as mock_get:
        # Mock responses for different pages
        responses = {
            "https://docs.example.com": """
                <html>
                    <body>
                        <a href="/page2">Page 2</a>
                        <a href="/page3">Page 3</a>
                    </body>
                </html>
            """,
            "https://docs.example.com/page2": "<html><body>Page 2</body></html>",
            "https://docs.example.com/page3": "<html><body>Page 3</body></html>",
        }

        def mock_get_response(url: str, **kwargs: Any) -> Mock:
            response = Mock()
            response.text = responses[url]
            return response

        mock_get.side_effect = mock_get_response

        processor.process_documentation(
            "test_project", "https://docs.example.com", max_depth=1
        )

        # Should store all three pages
        assert processor.storage.store_text.call_count == 3  # type: ignore[attr-defined]
        stored_urls = [
            call[0][2] for call in processor.storage.store_text.call_args_list  # type: ignore[attr-defined]
        ]
        assert sorted(stored_urls) == [
            "https://docs.example.com",
            "https://docs.example.com/page2",
            "https://docs.example.com/page3",
        ]


def test_process_with_external_links(processor: DocsProcessor) -> None:
    """Test that external links are not followed."""
    with patch.object(processor.crawler, "get") as mock_get:
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <a href="/internal">Internal Page</a>
                <a href="https://other-domain.com/page">External Page</a>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        processor.process_documentation(
            "test_project", "https://docs.example.com", max_depth=1
        )

        # Should only try to get the internal page
        assert mock_get.call_count == 2
        called_urls = [call[0][0] for call in mock_get.call_args_list]
        assert "https://other-domain.com/page" not in called_urls


def test_process_with_request_error(processor: DocsProcessor) -> None:
    """Test handling of request errors during crawling."""
    with patch.object(processor.crawler, "get") as mock_get:
        # First request succeeds, second fails
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <a href="/error-page">Error Page</a>
                <a href="/good-page">Good Page</a>
            </body>
        </html>
        """

        def mock_get_response(url: str, **kwargs: Any) -> Mock:
            if "error-page" in url:
                raise requests.RequestException("Failed to fetch")
            if "good-page" in url:
                response = Mock()
                response.text = "<html><body>Good Page</body></html>"
                return response
            return mock_response

        mock_get.side_effect = mock_get_response

        processor.process_documentation(
            "test_project", "https://docs.example.com", max_depth=1
        )

        # Should store the main page and good page, skip the error page
        assert processor.storage.store_text.call_count == 2  # type: ignore[attr-defined]
