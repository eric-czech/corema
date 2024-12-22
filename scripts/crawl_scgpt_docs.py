#!/usr/bin/env python3
"""Script to test crawling scGPT documentation from GitHub."""

import logging
from corema.storage import LocalStorage
from corema.processors.github import GitHubProcessor
from corema.processors.docs import DocsProcessor
from corema.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Project info
PROJECT_NAME = "scgpt"
GITHUB_URL = "https://github.com/bowang-lab/scGPT"


def main() -> None:
    """Main script function."""
    # Initialize storage and processors
    storage = LocalStorage()
    github_processor = GitHubProcessor(storage)
    docs_processor = DocsProcessor(storage)

    # Get max crawl depth from config
    config = get_config()
    max_depth = config.get("docs", {}).get("max_depth", 1)

    # Find documentation URLs from GitHub
    print(f"\nFinding documentation URLs for {GITHUB_URL}...")
    doc_urls = github_processor.infer_doc_urls(PROJECT_NAME, GITHUB_URL)

    if not doc_urls:
        print("No documentation URLs found.")
        return

    print(f"\nFound {len(doc_urls)} documentation URLs:")
    for url in doc_urls:
        print(f"- {url}")

    # Crawl each documentation URL
    for url in doc_urls:
        print(f"\nCrawling documentation at {url} (max_depth={max_depth})...")
        try:
            docs_processor.process_documentation(PROJECT_NAME, url, max_depth=max_depth)
            print(f"Successfully crawled {url}")
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    # Show where files were stored
    project_dir = storage.get_project_dir(PROJECT_NAME)
    docs_dir = project_dir / "docs"
    if docs_dir.exists():
        print(f"\nDocumentation files stored in: {docs_dir}")
        for path in docs_dir.rglob("*.html"):
            print(f"- {path.relative_to(storage.projects_dir)}")
    else:
        print("\nNo documentation files were stored.")


if __name__ == "__main__":
    main()
