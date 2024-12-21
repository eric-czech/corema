from urllib.parse import urlparse, urljoin
from typing import Optional

def normalize_github_url(url: str) -> str:
    """Normalize GitHub URLs to a consistent format."""
    parsed = urlparse(url)
    
    # Convert web URLs to git URLs
    if parsed.netloc == "github.com":
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"git@github.com:{path}.git"
    
    return url

def get_base_url(url: str) -> str:
    """Get base URL without path components."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain."""
    return urlparse(url1).netloc == urlparse(url2).netloc

def join_url(base: str, path: str) -> str:
    """Safely join base URL with path."""
    return urljoin(base, path) 