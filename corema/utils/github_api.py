"""GitHub API utilities."""

import logging
import requests
import re
from typing import Dict, Any, Literal
from ..config import get_config

logger = logging.getLogger(__name__)

# Type for URL normalization mode
GitHubURLMode = Literal["https", "ssh"]


def normalize_github_url(github_url: str, mode: str = "https") -> str:
    """Normalize a GitHub URL to either HTTPS or SSH format.

    Args:
        github_url: The GitHub URL to normalize.
        mode: The format to normalize to, either "https" or "ssh".

    Returns:
        The normalized URL.

    Raises:
        ValueError: If the URL is not a valid GitHub URL or the mode is invalid.
    """
    if mode not in ["https", "ssh"]:
        raise ValueError(f"Invalid mode: {mode}")

    owner, repo = extract_owner_repo(github_url)

    if mode == "https":
        return f"https://github.com/{owner}/{repo}"
    else:
        return f"git@github.com:{owner}/{repo}.git"


def extract_owner_repo(github_url: str) -> tuple[str, str]:
    """Extract the owner and repository name from a GitHub URL.

    Args:
        github_url: The GitHub URL to parse.

    Returns:
        A tuple of (owner, repository) strings.

    Raises:
        ValueError: If the URL is not a valid GitHub URL or the owner/repo cannot be extracted.
    """
    # Remove any hash fragments from the URL
    github_url = github_url.split("#")[0]

    # Match either HTTPS or SSH GitHub URLs
    https_pattern = (
        r"https://(?:raw\.)?github(?:usercontent)?\.com/([^/]+)/([^/#]+)(?:/.*)?$"
    )
    ssh_pattern = r"git@github\.com:([^/]+)/([^/#]+?)(?:\.git)?$"

    # Try HTTPS pattern first
    match = re.match(https_pattern, github_url)
    if match:
        return match.group(1), match.group(2).replace(".git", "")

    # Try SSH pattern
    match = re.match(ssh_pattern, github_url)
    if match:
        return match.group(1), match.group(2)

    raise ValueError(f"Could not extract owner/repo from URL: {github_url}")


def check_repo_exists(git_url: str) -> bool:
    """Check if a Git repository exists and is accessible.

    Args:
        git_url: URL of the Git repository (HTTPS or SSH)

    Returns:
        bool: True if repository exists and is accessible, False otherwise
    """
    try:
        owner, repo = extract_owner_repo(git_url)
    except ValueError:
        return False

    # Use GitHub API to check if repo exists
    token = get_config().get("GITHUB_API_TOKEN")
    if not token:
        logger.warning("No GitHub API token found, authentication may fail")
        return False

    headers = {"Authorization": f"token {token}"}
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers=headers,
    )
    return response.status_code == 200


def fetch_repo_metadata(github_url: str) -> Dict[str, Any]:
    """Fetch repository metadata from GitHub API.

    Args:
        github_url: GitHub repository URL (HTTPS or SSH)

    Returns:
        Repository metadata from GitHub API

    Raises:
        RuntimeError: If metadata cannot be fetched
    """
    try:
        owner, repo = extract_owner_repo(github_url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        logger.debug(f"Fetching metadata from GitHub API: {api_url}")

        headers: Dict[str, str] = {}
        token = get_config().get("GITHUB_API_TOKEN")
        if token:
            logger.debug("Using GitHub API token for authentication")
            headers["Authorization"] = f"token {token}"
        else:
            logger.warning("No GitHub API token found, requests may be rate limited")

        try:
            response = requests.get(api_url, headers=headers)
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Invalid response format from GitHub API: {api_url}")
            return data
        except Exception as e:
            raise RuntimeError(
                f"Error fetching repository metadata from {api_url}: {str(e)}"
            )
    except ValueError as e:
        raise RuntimeError(f"Error processing GitHub URL {github_url}: {str(e)}")
