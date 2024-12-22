"""Tests for paper processor."""

import pytest
from unittest.mock import patch, Mock
from datetime import date
from pathlib import Path
from typing import Any

from ..processors.paper import (
    PaperProcessor,
    PaperMetadata,
    Author,
    GitHubURLResponse,
    GitHubURL,
)
from ..storage import LocalStorage
from ..config import get_config
from .utils import check_openai_key


@pytest.fixture
def processor() -> PaperProcessor:
    """Create a PaperProcessor instance with a mocked storage."""
    storage = Mock(spec=LocalStorage)  # type: ignore
    return PaperProcessor(storage)


def test_infer_github_urls_empty_text(processor: PaperProcessor) -> None:
    """Test that empty text returns no URLs."""
    assert processor.infer_github_urls("") == []


def test_infer_github_urls_with_validation(
    processor: PaperProcessor, block_openai_api: Any
) -> None:
    """Test URL validation during extraction."""
    text = """
    Valid: https://github.com/valid/repo
    Invalid: https://github.com/invalid/repo
    Also valid: https://github.com/valid/another
    """

    def mock_check_repo_exists(url: str) -> bool:
        # Only return True for URLs that start with valid/
        return "/valid/" in url

    def mock_normalize_url(url: str, mode: str) -> str:
        return url  # Return URL unchanged for testing

    mock_response = GitHubURLResponse(
        github_urls=[
            GitHubURL(
                url="https://github.com/valid/repo",
                relevance="primary",
                confidence=0.95,
            ),
            GitHubURL(
                url="https://github.com/invalid/repo",
                relevance="primary",
                confidence=0.95,
            ),
            GitHubURL(
                url="https://github.com/valid/another",
                relevance="primary",
                confidence=0.95,
            ),
        ]
    )

    with patch(
        "corema.processors.paper.check_repo_exists", side_effect=mock_check_repo_exists
    ), patch(
        "corema.processors.paper.get_structured_output", return_value=mock_response
    ), patch(
        "corema.processors.paper.normalize_github_url", side_effect=mock_normalize_url
    ):
        urls = processor.infer_github_urls(text)
        assert sorted(urls) == sorted(
            [
                "https://github.com/valid/repo",
                "https://github.com/valid/another",
            ]
        )


@pytest.mark.llm
@pytest.mark.skipif(
    not check_openai_key(get_config().get("OPENAI_API_KEY")),
    reason="OPENAI_API_KEY not configured or invalid",
)
def test_github_url_extraction_llm(processor: PaperProcessor) -> None:
    """Test GitHub URL extraction using real LLM calls.

    This test requires a valid OpenAI API key and will make actual API calls.
    Run with pytest -m llm to include this test.
    """
    # Test case with explicit GitHub URL
    text_with_url = """
    Our implementation is available at https://github.com/openai/whisper.
    We used PyTorch for training and evaluation.
    """
    urls = processor.infer_github_urls(text_with_url)
    assert len(urls) == 1
    assert urls[0] == "https://github.com/openai/whisper"

    # Test case with no GitHub URL
    text_without_url = """
    We implemented our model using PyTorch and trained it on 8 A100 GPUs.
    The training took approximately 2 weeks to complete.
    """
    urls = processor.infer_github_urls(text_without_url)
    assert len(urls) == 0

    # Test case with reference URL that should be ignored
    text_with_reference = """
    Our implementation is available at https://github.com/huggingface/transformers.
    We compared our results with the baseline model from https://github.com/google/jax.
    """
    urls = processor.infer_github_urls(text_with_reference)
    assert len(urls) == 1
    assert urls[0] == "https://github.com/huggingface/transformers"


def test_extract_metadata_success(
    processor: PaperProcessor, block_openai_api: Any
) -> None:
    """Test successful metadata extraction."""
    text = """
    Title: Test Paper
    Authors: John Doe (University A), Jane Smith (University B, University C)
    DOI: 10.1234/test
    Journal: Test Journal
    Published: 2024-01-15
    """

    mock_metadata = {
        "title": "Test Paper",
        "doi": "10.1234/test",
        "publication_date": "2024-01-15",  # Pass as string, not date object
        "journal": "Test Journal",
        "authors": [
            {"name": "John Doe", "affiliations": ["University A"]},
            {"name": "Jane Smith", "affiliations": ["University B", "University C"]},
        ],
    }

    with patch(
        "corema.processors.paper.get_structured_output",
        return_value=PaperMetadata(**mock_metadata),
    ):
        metadata = processor.extract_metadata(text)
        assert metadata.title == "Test Paper"
        assert metadata.doi == "10.1234/test"
        assert metadata.publication_date == date(2024, 1, 15)
        assert metadata.journal == "Test Journal"
        assert len(metadata.authors) == 2
        assert metadata.authors[0].name == "John Doe"
        assert metadata.authors[0].affiliations == ["University A"]
        assert metadata.authors[1].name == "Jane Smith"
        assert metadata.authors[1].affiliations == ["University B", "University C"]


def test_extract_metadata_error(
    processor: PaperProcessor, block_openai_api: Any
) -> None:
    """Test metadata extraction error handling."""
    text = "Some paper text"

    with patch(
        "corema.processors.paper.get_structured_output",
        side_effect=Exception("API error"),
    ):
        metadata = processor.extract_metadata(text)
        assert metadata == PaperMetadata(authors=[])


def test_process_paper_success(
    processor: PaperProcessor, block_openai_api: Any
) -> None:
    """Test successful paper processing."""
    # Mock PDF content and response
    pdf_content = b"%PDF-1.4\n..."  # Minimal PDF content
    mock_response = Mock()
    mock_response.raw.read.return_value = pdf_content
    mock_response.headers = {}

    # Mock PDF text content
    text_content = """
    Title: Test Paper
    This is a test paper referencing https://github.com/test/repo
    """

    # Mock metadata
    mock_metadata = PaperMetadata(
        title="Test Paper",
        authors=[Author(name="Test Author", affiliations=["Test University"])],
    )

    # Mock LLM response
    mock_llm_response = GitHubURLResponse(
        github_urls=[
            GitHubURL(
                url="https://github.com/test/repo", relevance="primary", confidence=0.9
            ),
        ]
    )

    # Mock file path returned by store_binary
    mock_pdf_path = Path("mock/test.pdf")
    processor.storage.store_binary.return_value = mock_pdf_path  # type: ignore

    # Set up mocks
    with patch.object(
        processor, "_extract_text", return_value=text_content
    ), patch.object(processor.crawler, "get", return_value=mock_response), patch(
        "corema.processors.paper.check_repo_exists", return_value=True
    ), patch(
        "corema.processors.paper.normalize_github_url",
        return_value="https://github.com/test/repo",
    ), patch(
        "corema.processors.paper.get_structured_output",
        side_effect=[mock_metadata, mock_llm_response],
    ):
        # Process paper
        urls = processor.process_paper("test_project", "http://example.com/paper.pdf")

        # Verify results
        assert urls == ["https://github.com/test/repo"]
        processor.storage.store_binary.assert_called_once()  # type: ignore
        assert processor.storage.store_text.call_count == 2  # type: ignore # Once for metadata, once for text


def test_process_paper_error(processor: PaperProcessor) -> None:
    """Test paper processing error handling."""
    with patch.object(
        processor.crawler, "get", side_effect=Exception("Download failed")
    ):
        with pytest.raises(Exception):
            processor.process_paper("test_project", "http://example.com/paper.pdf")
