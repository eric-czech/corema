from typing import Set
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from ..storage import LocalStorage
from ..utils.url import Crawler

logger = logging.getLogger(__name__)


class DocsProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.visited: Set[str] = set()
        self.crawler = Crawler(
            rate_limit=2.0,  # 2 seconds between requests to same domain
            max_retries=5,
            backoff_factor=0.5,
            rotate_user_agents=True,
        )

    def process_documentation(
        self, project_name: str, doc_url: str, max_depth: int = 0
    ) -> None:
        """Crawl and store documentation site.

        Args:
            project_name: Name of the project
            doc_url: URL of the documentation to process
            max_depth: Maximum depth to crawl (0 means only the initial page, 1 means one level deep, etc.)
        """
        try:
            self.visited.clear()
            self._crawl_page(
                project_name, doc_url, current_depth=0, max_depth=max_depth
            )
        except Exception:
            logger.exception(f"Error processing documentation {doc_url}")
            raise

    def _crawl_page(
        self, project_name: str, url: str, current_depth: int, max_depth: int
    ) -> None:
        """Recursively crawl documentation pages.

        Args:
            project_name: Name of the project
            url: URL to crawl
            current_depth: Current depth in the crawl
            max_depth: Maximum depth to crawl
        """
        if url in self.visited:
            return

        self.visited.add(url)

        try:
            logger.info(f"Crawling {url!r}")
            response = self.crawler.get(url)

            # Store the page content
            self.storage.store_text(project_name, response.text, url, "docs", ".html")

            # Only continue crawling if we haven't reached max_depth
            if current_depth < max_depth:
                # Parse links and continue crawling
                soup = BeautifulSoup(response.text, "html.parser")
                base_domain = urlparse(url).netloc

                for link in soup.find_all("a"):
                    href = link.get("href")
                    if not href:
                        continue

                    # Ensure base URL has trailing slash for proper URL resolution
                    base_url = url if url.endswith("/") else url + "/"
                    full_url = urljoin(base_url, href)
                    if urlparse(full_url).netloc != base_domain:
                        continue

                    if full_url not in self.visited:
                        self._crawl_page(
                            project_name, full_url, current_depth + 1, max_depth
                        )

        except requests.RequestException as e:
            logger.warning(f"Failed to crawl {url} - skipping: {str(e)}")
        except Exception:
            logger.exception(f"Unexpected error crawling {url} - skipping")
