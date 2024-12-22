from typing import Dict, Any, List, Callable
import logging
from .storage import LocalStorage
from .utils.manifest import ManifestManager
from .processors.paper import PaperProcessor
from .processors.github import GitHubProcessor
from .processors.docs import DocsProcessor
from .processors.article import ArticleProcessor
from .config import get_config

# Set up logging
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, storage: LocalStorage):
        """
        Initialize the DataCollector.

        Args:
            storage: Storage backend for collected data
        """
        self.storage = storage
        self.manifest = ManifestManager()
        self.paper_processor = PaperProcessor(storage)
        self.github_processor = GitHubProcessor(storage)
        self.docs_processor = DocsProcessor(storage)
        self.article_processor = ArticleProcessor(storage)
        self.config = get_config()

    def _process_paper(self, project_name: str, paper_url: str) -> Dict[str, Any]:
        """Process a single paper URL."""
        try:
            logger.info(f"Processing paper: {paper_url}")
            github_urls = self.paper_processor.process_paper(project_name, paper_url)
            if github_urls:
                logger.info(
                    f"Found {len(github_urls)} GitHub URLs in paper: {github_urls}"
                )
            return {
                "success": True,
                "url": paper_url,
                "path": str(
                    self.storage.get_path_for_url(project_name, paper_url, "paper")
                ),
                "github_urls": github_urls or [],
            }
        except Exception:
            logger.exception(f"Failed to process paper {paper_url}")
            return {"success": False, "url": paper_url}

    def _process_github(self, project_name: str, github_url: str) -> Dict[str, Any]:
        """Process a single GitHub repository."""
        try:
            logger.info(f"Processing GitHub repository: {github_url}")
            doc_urls = self.github_processor.process_repository(
                project_name, github_url
            )
            if doc_urls:
                logger.info(f"Found {len(doc_urls)} documentation URLs from GitHub")
            return {
                "success": True,
                "url": github_url,
                "path": str(
                    self.storage.get_path_for_url(project_name, github_url, "github")
                ),
                "doc_urls": doc_urls or [],
            }
        except Exception:
            logger.exception(f"Failed to process GitHub repository {github_url}")
            return {"success": False, "url": github_url}

    def _process_docs(self, project_name: str, doc_url: str) -> Dict[str, Any]:
        """Process a single documentation URL."""
        try:
            logger.info(f"Processing documentation: {doc_url}")
            # Get max_depth from config, default to 1 if not specified
            max_depth = self.config.get("docs", {}).get("max_depth", 1)
            self.docs_processor.process_documentation(
                project_name, doc_url, max_depth=max_depth
            )
            return {
                "success": True,
                "url": doc_url,
                "path": str(
                    self.storage.get_path_for_url(project_name, doc_url, "docs")
                ),
            }
        except Exception:
            logger.exception(f"Failed to process documentation {doc_url}")
            return {"success": False, "url": doc_url}

    def _process_article(self, project_name: str, url: str) -> Dict[str, Any]:
        """Process a single article URL."""
        try:
            logger.info(f"Processing article: {url}")
            path = self.article_processor.process_article(project_name, url)
            if path:
                return {
                    "success": True,
                    "url": url,
                    "path": path,
                }
            return {"success": False, "url": url}
        except Exception:
            logger.exception(f"Failed to process article {url}")
            return {"success": False, "url": url}

    def _process_urls(
        self,
        project_name: str,
        urls: List[str],
        processor_func: Callable[[str, str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process a list of URLs synchronously."""
        if not urls:
            return []
        return [processor_func(project_name, url) for url in urls]

    def has_been_processed(self, project_id: str) -> bool:
        """Check if a project has already been processed.

        Args:
            project_id: ID of the project to check

        Returns:
            True if the project has been processed (manifest exists), False otherwise.
        """
        manifest_path = self.storage.get_project_dir(project_id) / "manifest.json"
        return manifest_path.exists()

    def process_project(self, project_data: Dict[str, Any]) -> None:
        """Process a single project's data collection.

        Args:
            project_data: Project data including URLs to process
        """
        project_name = project_data["project_name"]
        logger.info(f"Processing project: {project_name}")

        project_paths: Dict[str, List[Dict[str, str]]] = {
            "papers": [],
            "github_repos": [],
            "documentation": [],
            "articles": [],
        }

        # Track processed URLs to avoid duplicates
        processed_urls: Dict[str, set[str]] = {
            "papers": set(),
            "github_repos": set(),
            "documentation": set(),
            "articles": set(),
        }

        # Process papers
        paper_urls = project_data.get("paper_urls", [])
        if paper_urls:
            logger.info(f"Found {len(paper_urls)} papers to process")
            paper_results = self._process_urls(
                project_name, paper_urls, self._process_paper
            )
            for result in paper_results:
                if result["success"] and result["url"] not in processed_urls["papers"]:
                    processed_urls["papers"].add(result["url"])
                    project_paths["papers"].append(
                        {"url": result["url"], "path": result["path"]}
                    )
                    project_data.setdefault("github_urls", []).extend(
                        result["github_urls"]
                    )

        # Process GitHub repositories
        github_urls = project_data.get("github_urls", [])
        if github_urls:
            logger.info(f"Found {len(github_urls)} GitHub repositories to process")
            github_results = self._process_urls(
                project_name, github_urls, self._process_github
            )
            for result in github_results:
                if (
                    result["success"]
                    and result["url"] not in processed_urls["github_repos"]
                ):
                    processed_urls["github_repos"].add(result["url"])
                    project_paths["github_repos"].append(
                        {"url": result["url"], "path": result["path"]}
                    )
                    project_data.setdefault("doc_urls", []).extend(result["doc_urls"])

        # Process documentation
        doc_urls = project_data.get("doc_urls", [])
        if doc_urls:
            logger.info(f"Found {len(doc_urls)} documentation sites to process")
            doc_results = self._process_urls(project_name, doc_urls, self._process_docs)
            for result in doc_results:
                if (
                    result["success"]
                    and result["url"] not in processed_urls["documentation"]
                ):
                    processed_urls["documentation"].add(result["url"])
                    project_paths["documentation"].append(
                        {"url": result["url"], "path": result["path"]}
                    )

        # Process articles
        article_urls = project_data.get("article_urls", [])
        if article_urls:
            logger.info(f"Found {len(article_urls)} articles to process")
            article_results = self._process_urls(
                project_name, article_urls, self._process_article
            )
            for result in article_results:
                if (
                    result["success"]
                    and result["url"] not in processed_urls["articles"]
                ):
                    processed_urls["articles"].add(result["url"])
                    project_paths["articles"].append(
                        {"url": result["url"], "path": result["path"]}
                    )

        # Update manifest
        logger.info(f"Updating manifest for project: {project_name}")
        self.manifest.add_project(
            project_name, {"paths": project_paths, "metadata": project_data}
        )
