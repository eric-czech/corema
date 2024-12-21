import dramatiq
from typing import Set
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from ..storage import LocalStorage

class DocsProcessor:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.visited: Set[str] = set()

    @dramatiq.actor(max_retries=3, min_backoff=1000)
    def process_documentation(self, doc_url: str):
        """Crawl and store documentation site."""
        self.visited.clear()
        self._crawl_page(doc_url)

    def _crawl_page(self, url: str):
        """Recursively crawl documentation pages."""
        if url in self.visited:
            return
        
        self.visited.add(url)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Store the page content
            self.storage.store_text(
                response.text,
                url,
                ".html"
            )
            
            # Parse links and continue crawling
            soup = BeautifulSoup(response.text, 'html.parser')
            base_domain = urlparse(url).netloc
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href:
                    continue
                
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc != base_domain:
                    continue
                
                if full_url not in self.visited:
                    time.sleep(1)  # Be nice to the server
                    self._crawl_page(full_url)
                    
        except requests.RequestException:
            # Log error and continue
            pass 