import gzip
from pathlib import Path
from typing import List, Optional
import logging
from pdfminer.high_level import extract_text

from corema.storage import LocalStorage
from corema.utils.url import Crawler
from corema.utils.github_api import check_repo_exists, normalize_github_url
from corema.config import get_config, get_openai_params
from corema.utils.openai_api import get_structured_output
from corema.resources.paper_prompts import (
    GITHUB_URL_EXTRACTION_SYSTEM_PROMPT,
    GITHUB_URL_EXTRACTION_USER_PROMPT,
    PAPER_METADATA_SYSTEM_PROMPT,
    PAPER_METADATA_USER_PROMPT,
)
from pydantic import BaseModel, Field, field_validator
from datetime import date

logger = logging.getLogger(__name__)


class GitHubURL(BaseModel):
    """A GitHub URL with relevance and confidence scores."""

    url: str
    relevance: str = Field(
        description="Whether this is the primary repository for the model (primary) or just a reference/citation (reference)",
        pattern="^(primary|reference)$",
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0, le=1
    )


class GitHubURLResponse(BaseModel):
    """Response format for GitHub URL extraction."""

    github_urls: List[GitHubURL]


class Author(BaseModel):
    """An author with their affiliations."""

    name: str
    affiliations: List[str]


class PaperMetadata(BaseModel):
    """Metadata extracted from a paper."""

    title: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[date] = None
    journal: Optional[str] = None
    authors: List[Author]

    @field_validator("publication_date", mode="before")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[date]:
        """Convert and validate date string from LLM response."""
        if v is None:
            return None
        try:
            # Input should be YYYY-MM-DD from LLM
            year, month, day = map(int, v.split("-"))
            return date(year, month, day)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"Invalid date format or value: {str(e)}")


def combine_github_urls(
    a: GitHubURLResponse, b: GitHubURLResponse
) -> GitHubURLResponse:
    """Combine two GitHubURLResponse objects, keeping only unique URLs."""
    # Use a dict to track highest confidence for each URL
    url_map: dict[str, GitHubURL] = {}
    for item in a.github_urls + b.github_urls:
        if item.url not in url_map or item.confidence > url_map[item.url].confidence:
            url_map[item.url] = item

    return GitHubURLResponse(github_urls=list(url_map.values()))


def combine_metadata(a: PaperMetadata, b: PaperMetadata) -> PaperMetadata:
    """Combine metadata from two chunks, keeping the most complete information."""
    return PaperMetadata(
        doi=a.doi or b.doi,
        publication_date=a.publication_date or b.publication_date,
        journal=a.journal or b.journal,
        authors=list(
            {author.name: author for author in (a.authors + b.authors)}.values()
        ),
    )


class PaperProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.crawler = Crawler(
            rate_limit=2.0,  # Be nice to academic servers
            max_retries=5,
            backoff_factor=1.0,  # Longer backoff for academic servers
            rotate_user_agents=True,
        )
        self.config = get_config()

    def extract_metadata(self, text: str) -> PaperMetadata:
        """Extract metadata from paper text.

        Args:
            text: Text content to analyze (typically first page)

        Returns:
            Extracted paper metadata
        """
        try:
            # Get OpenAI parameters from paper processor config
            openai_params = get_openai_params(
                self.config.get("paper", {}).get("openai", {})
            )

            result = get_structured_output(
                system_prompt=PAPER_METADATA_SYSTEM_PROMPT,
                user_prompt=PAPER_METADATA_USER_PROMPT.format(text=text),
                response_model=PaperMetadata,
                reduce_op=combine_metadata,
                **openai_params,
            )

            return result

        except Exception:
            logger.exception("Error during metadata extraction")
            # Return empty metadata rather than failing
            return PaperMetadata(authors=[])

    def infer_github_urls(self, text: str) -> List[str]:
        """Extract GitHub URLs from text content.

        Args:
            text: Text content to analyze

        Returns:
            List of unique GitHub URLs found in the text
        """
        if not text:
            logger.debug("Empty text provided, returning empty list")
            return []

        try:
            logger.debug("Starting GitHub URL extraction")
            # Get OpenAI parameters from paper processor config
            openai_params = get_openai_params(
                self.config.get("paper", {}).get("openai", {})
            )
            logger.debug(f"Using OpenAI parameters: {openai_params}")

            logger.debug("Calling LLM for URL extraction")
            result = get_structured_output(
                system_prompt=GITHUB_URL_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=GITHUB_URL_EXTRACTION_USER_PROMPT.format(text=text),
                response_model=GitHubURLResponse,
                reduce_op=combine_github_urls,
                **openai_params,
            )
            logger.debug(f"LLM response: {result.model_dump_json(indent=2)}")

            # Filter for primary URLs with high confidence
            primary_urls = [
                item.url
                for item in result.github_urls
                if item.relevance == "primary" and item.confidence >= 0.8
            ]
            logger.debug(f"Primary URLs after filtering: {primary_urls}")

            # Normalize URLs
            normalized = []
            for url in primary_urls:
                try:
                    norm = normalize_github_url(url, mode="https")
                    logger.debug(f"Normalized URL {url} -> {norm}")
                    if not check_repo_exists(norm):
                        logger.warning(f"Skipping invalid GitHub URL: {norm}")
                        continue
                    normalized.append(norm)
                except ValueError:
                    logger.warning(f"Failed to normalize GitHub URL: {url}")
                    continue

            result_urls = sorted(list(set(normalized)))  # type: ignore
            logger.debug(f"Final URLs: {result_urls}")
            return result_urls

        except Exception:
            logger.exception("Error during LLM-based URL extraction")
            return []

    def process_paper(self, project_name: str, paper_url: str) -> List[str]:
        """Download and process a paper.

        Args:
            project_name: Name of the project
            paper_url: URL of the paper to analyze

        Returns:
            List of GitHub URLs found in the paper
        """
        try:
            logger.debug(f"Starting to process paper: {paper_url}")
            # Download PDF
            response = self.crawler.get(paper_url, stream=True)
            content = response.raw.read()

            # Check Content-Encoding header first
            if response.headers.get("Content-Encoding") == "gzip":
                logger.debug("Detected gzipped content from headers, decompressing")
                content = gzip.decompress(content)
            # Fall back to content inspection using magic bytes
            elif content.startswith(b"\x1f\x8b\x08"):
                logger.debug("Detected gzipped content from magic bytes, decompressing")
                content = gzip.decompress(content)

            # Store PDF
            pdf_path = self.storage.store_binary(
                project_name, content, paper_url, "paper", ".pdf"
            )
            logger.info(f"Stored PDF at {pdf_path}")

            # Extract metadata from first two pages
            logger.debug("Extracting metadata from first two pages")
            metadata_text = self._extract_text(pdf_path, page_numbers=[0, 1])
            metadata = self.extract_metadata(metadata_text)
            logger.info(
                f"Paper metadata summary: title='{metadata.title}', doi='{metadata.doi}', published={metadata.publication_date}, authors={len(metadata.authors)}"
            )
            logger.debug(f"Paper metadata: {metadata.model_dump_json(indent=2)}")

            # Store metadata using same URL hash as PDF/text
            self.storage.store_text(
                project_name,
                metadata.model_dump_json(indent=2),
                paper_url,
                "paper",
                ".json",
            )
            logger.info("Stored paper metadata")

            # Extract and store full text
            full_text = self._extract_text(pdf_path)
            text_path = self.storage.store_text(
                project_name, full_text, paper_url, "paper", ".txt"
            )
            logger.info(f"Stored text at {text_path}")

            # Find GitHub URLs in the text
            logger.debug("Starting GitHub URL extraction")
            github_urls = self.infer_github_urls(full_text)
            if github_urls:
                logger.info(
                    f"Found {len(github_urls)} GitHub URLs in paper {paper_url}"
                )
            else:
                logger.warning("No GitHub URLs found in paper")
            return github_urls

        except Exception:
            logger.exception(f"Error processing paper {paper_url}")
            raise

    def _extract_text(
        self, pdf_path: Path, page_numbers: List[int] | None = None
    ) -> str:
        """Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            page_numbers: List of page numbers to extract (0-based). If None, extracts all pages.

        Returns:
            Extracted text content
        """
        text = extract_text(str(pdf_path), page_numbers=page_numbers)  # type: ignore[no-any-return]
        if text is None:
            return ""
        return str(text)
