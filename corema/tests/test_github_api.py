"""Tests for GitHub API utilities."""

import pytest
from unittest.mock import patch, Mock
from typing import Tuple

from corema.utils.github_api import (
    check_repo_exists,
    fetch_repo_metadata,
    normalize_github_url,
    extract_owner_repo,
)
from corema.config import get_config

# Load config to ensure token is available
get_config()


@pytest.mark.parametrize(
    "git_url,expected",
    [
        ("git@github.com:openai/gpt-3.git", True),  # Known public repo (SSH)
        ("https://github.com/openai/gpt-3", True),  # Known public repo (HTTPS)
        ("git@github.com:nonexistent/repo.git", False),  # Non-existent repo (SSH)
        ("https://github.com/nonexistent/repo", False),  # Non-existent repo (HTTPS)
        ("git@github.com:invalid-url", False),  # Invalid URL format (SSH)
        ("https://github.com/invalid-url", False),  # Invalid URL format (HTTPS)
    ],
)
def test_check_repo_exists(git_url: str, expected: bool) -> None:
    """Test checking if a repository exists.

    Args:
        git_url: Git URL to test (SSH or HTTPS)
        expected: Whether the repo should exist
    """
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200 if expected else 404
        mock_get.return_value = mock_response
        assert check_repo_exists(git_url) == expected


def test_check_repo_exists_with_mock() -> None:
    """Test check_repo_exists with mocked API responses."""
    with patch("requests.get") as mock_get:
        # Test successful cases
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert check_repo_exists("git@github.com:some/repo.git") is True
        assert check_repo_exists("https://github.com/some/repo") is True

        # Test failure cases
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        assert check_repo_exists("git@github.com:some/repo.git") is False
        assert check_repo_exists("https://github.com/some/repo") is False


@pytest.mark.parametrize(
    "github_url",
    [
        "git@github.com:openai/gpt-3.git",  # SSH URL
        "https://github.com/openai/gpt-3",  # HTTPS URL
    ],
)
def test_fetch_repo_metadata(github_url: str) -> None:
    """Test fetching repository metadata.

    Args:
        github_url: GitHub URL to test
    """
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "gpt-3",
            "owner": {"login": "openai"},
            "description": "Test repo",
        }
        mock_get.return_value = mock_response
        metadata = fetch_repo_metadata(github_url)
        assert metadata["name"] == "gpt-3"
        assert metadata["owner"]["login"] == "openai"


def test_fetch_repo_metadata_errors() -> None:
    """Test error handling in fetch_repo_metadata."""
    with patch("requests.get") as mock_get:
        # Test invalid URL
        with pytest.raises(RuntimeError) as exc_info:
            fetch_repo_metadata("invalid-url")
        assert "Could not extract owner/repo from URL" in str(exc_info.value)

        # Test API error
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        with pytest.raises(RuntimeError) as exc_info:
            fetch_repo_metadata("https://github.com/some/repo")
        assert "Error fetching repository metadata" in str(exc_info.value)


@pytest.mark.parametrize(
    "url,expected",
    [
        # Basic URLs
        ("git@github.com:username/repo.git", "https://github.com/username/repo"),
        ("https://github.com/username/repo", "https://github.com/username/repo"),
        ("https://github.com/username/repo.git", "https://github.com/username/repo"),
        # URLs with special characters
        (
            "git@github.com:org/repo-with-dashes.git",
            "https://github.com/org/repo-with-dashes",
        ),
        ("https://github.com/user/repo.name", "https://github.com/user/repo.name"),
        (
            "https://github.com/user-name/repo-name",
            "https://github.com/user-name/repo-name",
        ),
        # URLs with paths and fragments
        (
            "https://github.com/user/repo/blob/main/README.md",
            "https://github.com/user/repo",
        ),
        ("https://github.com/org/project#readme", "https://github.com/org/project"),
        # Raw URLs
        (
            "https://raw.githubusercontent.com/user/repo/main/file.txt",
            "https://github.com/user/repo",
        ),
    ],
)
def test_normalize_github_url_https(url: str, expected: str) -> None:
    """Test normalizing GitHub URLs to HTTPS format.

    Args:
        url: Input URL
        expected: Expected normalized URL
    """
    assert normalize_github_url(url, mode="https") == expected


@pytest.mark.parametrize(
    "input_url,expected_url",
    [
        (
            "git@github.com:username/repo.git",
            "git@github.com:username/repo.git",
        ),
        (
            "https://github.com/username/repo",
            "git@github.com:username/repo.git",
        ),
        (
            "https://github.com/username/repo.git",
            "git@github.com:username/repo.git",
        ),
        (
            "git@github.com:org/repo-with-dashes.git",
            "git@github.com:org/repo-with-dashes.git",
        ),
    ],
)
def test_normalize_github_url_ssh(input_url: str, expected_url: str) -> None:
    """Test normalizing GitHub URLs to SSH format.

    Args:
        input_url: Input URL in any format
        expected_url: Expected normalized SSH URL
    """
    assert normalize_github_url(input_url, mode="ssh") == expected_url


def test_normalize_github_url_invalid() -> None:
    """Test normalizing invalid GitHub URLs."""
    with pytest.raises(ValueError) as exc_info:
        normalize_github_url("invalid-url", mode="https")
    assert "Could not extract owner/repo from URL" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        normalize_github_url("invalid-url", mode="ssh")
    assert "Could not extract owner/repo from URL" in str(exc_info.value)


@pytest.mark.parametrize(
    "url,expected",
    [
        # Standard HTTPS URLs
        (
            "https://github.com/user/repo",
            ("user", "repo"),
        ),
        (
            "https://github.com/org/repo-with-dashes",
            ("org", "repo-with-dashes"),
        ),
        # SSH URLs
        (
            "git@github.com:user/repo.git",
            ("user", "repo"),
        ),
        (
            "git@github.com:org/repo.with.dots.git",
            ("org", "repo.with.dots"),
        ),
        # Raw GitHub URLs
        (
            "https://raw.githubusercontent.com/user/repo/main/file.txt",
            ("user", "repo"),
        ),
        (
            "https://raw.githubusercontent.com/org/repo-name/branch/path/file",
            ("org", "repo-name"),
        ),
    ],
)
def test_extract_owner_repo(url: str, expected: Tuple[str, str]) -> None:
    """Test extracting owner and repo from various GitHub URL formats.

    Args:
        url: GitHub URL to test
        expected: Expected (owner, repo) tuple
    """
    assert extract_owner_repo(url) == expected


@pytest.mark.parametrize(
    "invalid_url",
    [
        "https://not-github.com/user/repo",  # Wrong domain
        "https://github.com/incomplete",  # Missing repo
        "git@gitlab.com:user/repo.git",  # Wrong git domain
        "https://raw.githubusercontent.com/incomplete",  # Incomplete raw URL
        "invalid-url",  # Not a URL at all
    ],
)
def test_extract_owner_repo_invalid(invalid_url: str) -> None:
    """Test error handling for invalid GitHub URLs.

    Args:
        invalid_url: Invalid URL to test
    """
    with pytest.raises(ValueError) as exc_info:
        extract_owner_repo(invalid_url)
    assert "Could not extract owner/repo from URL" in str(exc_info.value)
