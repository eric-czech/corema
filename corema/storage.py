import logging
from pathlib import Path
import shutil
import json
import hashlib
from typing import Dict, Any, Literal, Union, BinaryIO, TextIO

from corema.utils.names import get_project_id
from corema.config import get_config

logger = logging.getLogger(__name__)

# Type for data categories
DataType = Literal["paper", "github", "docs", "article"]


class LocalStorage:
    def __init__(self, base_path: str | None = None):
        config = get_config()
        storage_config = config.get("storage", {})

        self.base_path = Path(base_path or storage_config.get("base_path", "./data"))
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create projects directory
        self.projects_dir = self.base_path / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized local storage at {self.base_path}")

    def get_project_dir(self, project_name: str) -> Path:
        """Get the project directory, creating it if it doesn't exist."""
        project_id = get_project_id(project_name)
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def get_path_for_url(
        self, project_name: str, url: str, data_type: DataType
    ) -> Path:
        """Generate a file path based on URL hash."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        path = self.get_project_dir(project_name) / data_type / url_hash
        logger.debug(f"Generated path for URL {url}: {path}")
        return path

    def store_binary(
        self,
        project_name: str,
        data: Union[bytes, BinaryIO],
        url: str,
        data_type: DataType,
        suffix: str = "",
    ) -> Path:
        """Store binary data with optional suffix."""
        try:
            path = self.get_path_for_url(project_name, url, data_type)
            if suffix:
                path = path.with_suffix(suffix)

            logger.info(f"Storing binary data at {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                with path.open("wb") as f:
                    shutil.copyfileobj(data, f)
                logger.debug(f"Successfully stored binary data at {path}")

            return path
        except Exception:
            logger.exception(f"Failed to store binary data for URL {url}")
            raise

    def store_text(
        self,
        project_name: str,
        text: Union[str, TextIO],
        url: str,
        data_type: DataType,
        suffix: str = "",
    ) -> Path:
        """Store text data with optional suffix."""
        try:
            path = self.get_path_for_url(project_name, url, data_type)
            if suffix:
                path = path.with_suffix(suffix)

            logger.info(f"Storing text data at {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(text, str):
                path.write_text(text)
            else:
                with path.open("w") as f:
                    shutil.copyfileobj(text, f)
                logger.debug(f"Successfully stored text data at {path}")

            return path
        except Exception:
            logger.exception(f"Failed to store text data for URL {url}")
            raise

    def store_manifest(self, project_name: str, data: Dict[str, Any]) -> Path:
        """Store project manifest data.

        Args:
            project_name: Name of the project
            data: Project manifest data to store

        Returns:
            Path where the manifest was stored
        """
        try:
            project_dir = self.get_project_dir(project_name)
            manifest_path = project_dir / "manifest.json"

            logger.info(f"Storing manifest at {manifest_path}")
            manifest_json = json.dumps(data, indent=2)
            manifest_path.write_text(manifest_json)
            logger.debug(f"Successfully stored manifest at {manifest_path}")

            return manifest_path
        except Exception:
            logger.exception(f"Failed to store manifest for project {project_name}")
            raise

    def load_manifest(self, project_name: str) -> Dict[str, Any] | None:
        """Load project manifest data.

        Args:
            project_name: Name of the project

        Returns:
            Project manifest data if it exists, None otherwise
        """
        try:
            project_dir = self.get_project_dir(project_name)
            manifest_path = project_dir / "manifest.json"

            if manifest_path.exists():
                logger.debug(f"Loading manifest from {manifest_path}")
                with open(manifest_path, "r") as f:
                    data: Dict[str, Any] = json.load(f)
                    return data
            return None
        except Exception:
            logger.exception(f"Failed to load manifest for project {project_name}")
            raise
