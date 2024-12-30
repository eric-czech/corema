"""Tests for GitHub repository processor."""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
from typing import List, Dict, Any
import json

from corema.processors.github import GitHubProcessor
from corema.storage import LocalStorage


@pytest.fixture
def processor() -> GitHubProcessor:
    """Create a GitHubProcessor instance with a mocked storage."""
    storage = Mock(spec=LocalStorage)
    return GitHubProcessor(storage)


def test_process_repository_success(processor: GitHubProcessor) -> None:
    """Test successful repository processing."""
    # Mock file path and URL
    mock_path = Path("mock/repo")
    github_url = "https://github.com/test-org/test-repo"
    processor.storage.get_path_for_url.return_value = mock_path  # type: ignore[attr-defined]

    # Mock repository data
    mock_data: Dict[str, Any] = {
        "name": "test-repo",
        "description": "A test repository",
        "stargazers_count": 100,
        "forks_count": 50,
        "language": "Python",
        "topics": ["machine-learning", "deep-learning"],
        "created_at": "2021-01-01T00:00:00Z",
        "updated_at": "2021-12-31T00:00:00Z",
        "owner": {"login": "test-org"},
    }

    # Set up mocks
    with patch(
        "corema.processors.github.fetch_repo_metadata", return_value=mock_data
    ), patch(
        "corema.processors.github.extract_owner_repo",
        return_value=("test-org", "test-repo"),
    ), patch(
        "corema.processors.github.check_repo_exists", return_value=True
    ), patch(
        "git.Repo.clone_from"
    ) as mock_clone, patch.object(
        processor, "infer_doc_urls", return_value=[]
    ):
        # Process repository and verify doc URLs
        doc_urls = processor.process_repository("test_project", github_url)
        assert doc_urls == []

        # Verify repository was cloned
        mock_clone.assert_called_once_with(
            "git@github.com:test-org/test-repo.git", mock_path
        )

        # Verify metadata was stored
        processor.storage.store_text.assert_called_once()  # type: ignore[attr-defined]
        stored_metadata = json.loads(processor.storage.store_text.call_args[0][1])  # type: ignore[attr-defined]
        assert stored_metadata["name"] == "test-repo"
        assert stored_metadata["description"] == "A test repository"
        assert stored_metadata["stargazers_count"] == 100
        assert stored_metadata["forks_count"] == 50
        assert stored_metadata["language"] == "Python"
        assert stored_metadata["topics"] == ["machine-learning", "deep-learning"]
        assert stored_metadata["created_at"] == "2021-01-01T00:00:00Z"
        assert stored_metadata["updated_at"] == "2021-12-31T00:00:00Z"
        assert stored_metadata["owner"]["login"] == "test-org"


def test_process_repository_nonexistent(processor: GitHubProcessor) -> None:
    """Test processing a repository that doesn't exist.

    Args:
        processor: GitHubProcessor fixture
    """
    github_url = "https://github.com/org/nonexistent"
    processor.storage.get_path_for_url.return_value = Path("mock/path")  # type: ignore[attr-defined, assignment]

    with patch("corema.processors.github.check_repo_exists", return_value=False):
        doc_urls = processor.process_repository("test_project", github_url)
        assert doc_urls == []


@pytest.mark.parametrize(
    "readme_content,github_url,expected_urls",
    [
        # Only GitHub Pages
        (
            """
            # Project Title
            Some content
            """,
            "https://github.com/org/myrepo",
            ["https://org.github.io/myrepo"],
        ),
        # Only ReadTheDocs
        (
            """
            # Project Title
            Some content
            """,
            "https://github.com/org/myrepo",
            ["https://myrepo.readthedocs.io/en/latest"],
        ),
        # Both GitHub Pages and ReadTheDocs
        (
            """
            # Project Title
            Some content
            """,
            "https://github.com/org/myrepo",
            [
                "https://myrepo.readthedocs.io/en/latest",
                "https://org.github.io/myrepo",
            ],
        ),
        # Neither GitHub Pages nor ReadTheDocs
        (
            """
            # Project Title
            Some content
            """,
            "https://github.com/org/myrepo",
            [],
        ),
    ],
)
def test_doc_urls_from_readme(
    readme_content: str,
    github_url: str,
    expected_urls: List[str],
    processor: GitHubProcessor,
    tmp_path: Path,
) -> None:
    """Test extracting documentation URLs from README content."""
    # Create mock repository directory
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    readme_path = repo_path / "README.md"
    readme_path.write_text(readme_content)

    # Mock storage to return our test path
    processor.storage.get_path_for_url.return_value = repo_path  # type: ignore[attr-defined]

    # Mock URL validation based on expected URLs
    def mock_head(url: str, allow_redirects: bool = False) -> Mock:
        response = Mock()
        response.status_code = 200 if url in expected_urls else 404
        return response

    with patch.object(processor.crawler, "head", side_effect=mock_head):
        doc_urls = processor.infer_doc_urls("test_project", github_url)
        assert sorted(doc_urls) == sorted(expected_urls)
