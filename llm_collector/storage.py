from pathlib import Path
import hashlib
import shutil
from typing import Union, BinaryIO, TextIO

class LocalStorage:
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_path_for_url(self, url: str) -> Path:
        """Generate a file path based on URL hash."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.base_path / url_hash

    def store_binary(self, data: Union[bytes, BinaryIO], url: str, suffix: str = "") -> Path:
        """Store binary data with optional suffix."""
        path = self.get_path_for_url(url)
        if suffix:
            path = path.with_suffix(suffix)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            with path.open('wb') as f:
                shutil.copyfileobj(data, f)
        
        return path

    def store_text(self, text: Union[str, TextIO], url: str, suffix: str = "") -> Path:
        """Store text data with optional suffix."""
        path = self.get_path_for_url(url)
        if suffix:
            path = path.with_suffix(suffix)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(text, str):
            path.write_text(text)
        else:
            with path.open('w') as f:
                shutil.copyfileobj(text, f)
        
        return path 