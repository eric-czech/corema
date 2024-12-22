# type: ignore
import pytest
from unittest.mock import Mock, create_autospec, patch
import requests
from pathlib import Path

from ..processors.article import ArticleProcessor
from ..storage import LocalStorage


@pytest.fixture
def storage() -> Mock:
    """Create a mock storage instance."""
    mock = create_autospec(LocalStorage)  # type: ignore[no-any-return]
    return mock


@pytest.fixture
def processor(storage: Mock) -> ArticleProcessor:
    """Create an ArticleProcessor instance with mocked storage."""
    return ArticleProcessor(storage)


def test_process_html_article(processor: ArticleProcessor) -> None:
    """Test processing an HTML article."""
    with patch.object(processor.crawler, "get") as mock_get:
        mock_response = Mock()
        mock_response.text = "<html>Test content</html>"
        mock_response.content = b"<html>Test content</html>"
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_get.return_value = mock_response

        processor.storage.store_text.return_value = Path("data/test/article/123.html")  # type: ignore[attr-defined]

        path = processor.process_article("test_project", "https://example.com/article")

        assert path == "data/test/article/123.html"
        processor.storage.store_text.assert_called_once_with(  # type: ignore[attr-defined]
            "test_project",
            "<html>Test content</html>",
            "https://example.com/article",
            "article",
            ".html",
        )


def test_process_pdf_article(processor: ArticleProcessor) -> None:
    """Test processing a PDF article."""
    with patch.object(processor.crawler, "get") as mock_get:
        mock_response = Mock()
        mock_response.content = b"%PDF-1.4 test content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_get.return_value = mock_response

        processor.storage.store_binary.return_value = Path("data/test/article/123.pdf")  # type: ignore[attr-defined]

        path = processor.process_article(
            "test_project", "https://example.com/article.pdf"
        )

        assert path == "data/test/article/123.pdf"
        processor.storage.store_binary.assert_called_once_with(  # type: ignore[attr-defined]
            "test_project",
            b"%PDF-1.4 test content",
            "https://example.com/article.pdf",
            "article",
            ".pdf",
        )


def test_process_article_request_error(processor: ArticleProcessor) -> None:
    """Test handling of request errors."""
    with patch.object(processor.crawler, "get") as mock_get:
        mock_get.side_effect = requests.RequestException("Failed to fetch")

        path = processor.process_article("test_project", "https://example.com/error")

        assert path is None
        processor.storage.store_text.assert_not_called()  # type: ignore[attr-defined]
        processor.storage.store_binary.assert_not_called()  # type: ignore[attr-defined]


def test_process_article_retry_success(processor: ArticleProcessor) -> None:
    """Test successful retry after initial failure."""
    with patch.object(processor.crawler.session, "get") as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Test content"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = Mock()

        # Configure mock to fail twice then succeed
        mock_get.side_effect = [
            requests.RequestException("Temporary error"),
            requests.RequestException("Temporary error"),
            mock_response,
        ]

        # Mock sleep to avoid waiting in tests
        with patch("time.sleep"):
            processor.storage.store_text.return_value = Path("data/test/article/123.html")  # type: ignore[attr-defined]

            path = processor.process_article(
                "test_project", "https://example.com/retry"
            )

            assert path == "data/test/article/123.html"
            processor.storage.store_text.assert_called_once_with(  # type: ignore[attr-defined]
                "test_project",
                "Test content",
                "https://example.com/retry",
                "article",
                ".html",
            )
            assert mock_get.call_count == 3  # Two failures and one success
