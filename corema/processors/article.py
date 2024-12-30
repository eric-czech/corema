import logging
from typing import Optional
import requests

from corema.storage import LocalStorage
from corema.utils.url import Crawler

logger = logging.getLogger(__name__)


class ArticleProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.crawler = Crawler(
            rate_limit=2.0,  # 2 seconds between requests to same domain
            max_retries=5,
            backoff_factor=0.5,
            rotate_user_agents=True,
        )

    def process_article(self, project_name: str, article_url: str) -> Optional[str]:
        """Process and store an article from any URL.

        Args:
            project_name: Name of the project
            article_url: URL of the article to process

        Returns:
            Path where the article was stored, or None if processing failed
        """
        try:
            logger.info(f"Processing article: {article_url}")
            response = self.crawler.get(article_url)

            # Try to determine content type from headers
            content_type = response.headers.get("content-type", "").lower()

            if "application/pdf" in content_type:
                # Store as PDF
                path = self.storage.store_binary(
                    project_name, response.content, article_url, "article", ".pdf"
                )
            else:
                # Assume HTML/text otherwise
                path = self.storage.store_text(
                    project_name, response.text, article_url, "article", ".html"
                )

            logger.info(f"Successfully stored article at {path}")
            return str(path)

        except requests.RequestException:
            logger.warning(f"Failed to process article {article_url}", exc_info=True)
            return None
        except Exception:
            logger.exception(f"Unexpected error processing article {article_url}")
            return None
