import git
import requests
from typing import List
import logging
from ..storage import LocalStorage
from ..utils.url import Crawler
from ..utils.github_api import (
    check_repo_exists,
    fetch_repo_metadata,
    extract_owner_repo,
    normalize_github_url,
)
from ..config import get_config
import json

logger = logging.getLogger(__name__)


class GitHubProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.config = get_config()
        github_config = self.config.get("github", {})

        self.api_base = github_config.get("api_base", "https://api.github.com")
        self.crawler = Crawler(
            rate_limit=github_config.get(
                "rate_limit", 1.0
            ),  # Respect GitHub API rate limits
            max_retries=github_config.get("max_retries", 3),
            backoff_factor=github_config.get("backoff_factor", 0.5),
            rotate_user_agents=False,  # GitHub API doesn't need rotating user agents
        )

    def process_repository(self, project_name: str, github_url: str) -> List[str]:
        """Clone and process a GitHub repository.

        Args:
            project_name: Name of the project
            github_url: URL of the GitHub repository

        Returns:
            List of inferred documentation URLs
        """
        try:
            # Normalize URL for storage and metadata
            github_https_url = normalize_github_url(github_url, mode="https")
            repo_path = self.storage.get_path_for_url(
                project_name, github_https_url, "github"
            )

            # Convert to SSH URL for git operations
            github_ssh_url = normalize_github_url(github_url, mode="ssh")
            logger.debug(f"Using Git URL for clone: {github_ssh_url}")

            # Check if repository exists
            if not check_repo_exists(github_https_url):
                logger.warning(
                    f"Repository does not exist or is not accessible: {github_https_url}"
                )
                return []

            # Clone repository
            logger.info(f"Cloning repository to {repo_path}")
            try:
                git.Repo.clone_from(github_ssh_url, repo_path)
                logger.info("Repository cloned successfully")
            except git.GitCommandError:
                logger.exception(f"Failed to clone repository {github_ssh_url}")
                raise

            # Store repository metadata
            logger.info("Fetching repository metadata")
            try:
                metadata = fetch_repo_metadata(github_https_url)
                if metadata:
                    metadata_json = json.dumps(metadata, indent=2)
                    self.storage.store_text(
                        project_name,
                        metadata_json,
                        f"{github_https_url}#metadata",
                        "github",
                        ".json",
                    )
                    logger.info("Repository metadata stored successfully")
            except Exception:
                logger.exception(
                    f"Failed to fetch or store repository metadata for {github_https_url}"
                )
                raise

            # Infer documentation URLs
            doc_urls = self.infer_doc_urls(project_name, github_https_url)
            return doc_urls or []

        except Exception:
            logger.exception(f"Error processing repository {github_url}")
            raise

    def infer_doc_urls(self, project_name: str, github_url: str) -> List[str]:
        """Infer documentation URLs from GitHub repository.

        Only checks for GitHub Pages and ReadTheDocs URLs.
        """
        doc_urls: List[str] = []

        try:
            owner, repo = extract_owner_repo(github_url)
        except ValueError:
            return doc_urls

        # Check for GitHub Pages
        pages_url = f"https://{owner}.github.io/{repo}"
        try:
            response = self.crawler.head(pages_url, allow_redirects=True)
            if response.status_code == 200:
                doc_urls.append(pages_url)
        except requests.RequestException:
            pass

        # Check for ReadTheDocs
        rtd_url = f"https://{repo}.readthedocs.io/en/latest"
        try:
            response = self.crawler.head(rtd_url, allow_redirects=True)
            if response.status_code == 200:
                doc_urls.append(rtd_url)
        except requests.RequestException:
            pass

        return sorted(doc_urls)
