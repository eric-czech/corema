import dramatiq
import requests
from pathlib import Path
import pdfplumber
from typing import List
from ..storage import LocalStorage

class PaperProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage

    @dramatiq.actor(max_retries=3, min_backoff=1000)
    def process_paper(self, paper_url: str):
        """Download and process a paper."""
        # Download PDF
        response = requests.get(paper_url, stream=True)
        response.raise_for_status()
        
        # Store PDF
        pdf_path = self.storage.store_binary(response.raw, paper_url, ".pdf")
        
        # Extract text
        text = self._extract_text(pdf_path)
        self.storage.store_text(text, paper_url, ".txt")

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(text)

    def infer_github_urls(self, paper_url: str) -> List[str]:
        """Infer GitHub URLs from paper content."""
        # Implementation would parse the paper text to find GitHub URLs
        # This is a placeholder
        return [] 