import dramatiq
import git
import requests
from pathlib import Path
from typing import List
import re
from ..storage import LocalStorage
from ..config import Config

class GitHubProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.api_base = "https://api.github.com"

    @dramatiq.actor(max_retries=3, min_backoff=1000)
    def process_repository(self, github_url: str):
        """Clone and process a GitHub repository."""
        repo_path = self.storage.get_path_for_url(github_url)
        
        # Convert HTTPS URL to git URL if needed
        if github_url.startswith("https://github.com"):
            github_url = github_url.replace("https://github.com", "git@github.com:", 1)
            if not github_url.endswith(".git"):
                github_url += ".git"

        # Clone repository
        git.Repo.clone_from(github_url, repo_path)
        
        # Store repository metadata
        metadata = self._fetch_repo_metadata(github_url)
        self.storage.store_text(
            metadata, 
            f"{github_url}#metadata",
            ".json"
        )

    def _fetch_repo_metadata(self, github_url: str) -> str:
        """Fetch repository metadata from GitHub API."""
        # Extract owner/repo from URL
        match = re.search(r"github\.com[:/]([^/]+)/([^/\.]+)", github_url)
        if not match:
            return "{}"
        
        owner, repo = match.groups()
        api_url = f"{self.api_base}/repos/{owner}/{repo}"
        
        headers = {}
        token = Config().get("GITHUB_API_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.text

    def infer_doc_urls(self, github_url: str) -> List[str]:
        """Infer documentation URLs from GitHub repository."""
        doc_urls = []
        
        # Extract owner/repo from URL
        match = re.search(r"github\.com[:/]([^/]+)/([^/\.]+)", github_url)
        if not match:
            return doc_urls
        
        owner, repo = match.groups()
        
        # Check for GitHub Pages
        pages_url = f"https://{owner}.github.io/{repo}"
        response = requests.head(pages_url)
        if response.status_code == 200:
            doc_urls.append(pages_url)
        
        # Check for ReadTheDocs
        rtd_url = f"https://{repo}.readthedocs.io"
        response = requests.head(rtd_url)
        if response.status_code == 200:
            doc_urls.append(rtd_url)
        
        return doc_urls 