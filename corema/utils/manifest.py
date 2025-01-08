from typing import Dict, Any, TypedDict, List, Iterator
import logging
from corema.utils.names import get_project_id
from corema.storage import LocalStorage

# Set up logging
logger = logging.getLogger(__name__)


class PathEntry(TypedDict):
    url: str
    path: str


class PathMapping(TypedDict):
    papers: List[PathEntry]
    github_repos: List[PathEntry]
    documentation: List[PathEntry]
    articles: List[PathEntry]


class ProjectMetadata(TypedDict):
    project_id: str
    project_name: str
    paths: PathMapping
    metadata: Dict[str, Any]


class ManifestManager:
    def __init__(self, base_path: str | None = None):
        """Initialize the manifest manager.

        Args:
            base_path: Base directory for data storage. If None, uses config value.
        """
        self.storage = LocalStorage(base_path)
        logger.info(f"Using base directory: {self.storage.base_path}")

    def _load_project(self, project_id: str) -> ProjectMetadata:
        """Load a project's manifest file.

        Args:
            project_id: Project ID

        Returns:
            Project metadata
        """
        manifest = self.storage.load_manifest(project_id)
        if manifest is not None:
            metadata = manifest.get("metadata", {})
            return ProjectMetadata(
                project_id=project_id,
                project_name=metadata.get("project_name", project_id),
                paths=manifest.get(
                    "paths",
                    {
                        "papers": [],
                        "github_repos": [],
                        "documentation": [],
                        "articles": [],
                    },
                ),
                metadata=metadata,
            )
        return ProjectMetadata(
            project_id=project_id,
            project_name=project_id,
            paths={
                "papers": [],
                "github_repos": [],
                "documentation": [],
                "articles": [],
            },
            metadata={},
        )

    def _save_project(self, project_id: str, data: ProjectMetadata) -> None:
        """Save a project's manifest file.

        Args:
            project_id: Project ID
            data: Project metadata to save
        """
        self.storage.store_manifest(project_id, dict(data))
        logger.info(f"Saved manifest for project {project_id}")

    def add_project(self, project_name: str, data: Dict[str, Any]) -> None:
        """Add or update a project in the manifest.

        Args:
            project_name: Name of the project
            data: Project data including paths and metadata
        """
        project_id = get_project_id(project_name)
        paths = data.get("paths", {})
        metadata = data.get("metadata", {})
        metadata["project_name"] = project_name

        # Ensure all path lists exist
        paths.setdefault("papers", [])
        paths.setdefault("github_repos", [])
        paths.setdefault("documentation", [])
        paths.setdefault("articles", [])

        project_data = ProjectMetadata(
            project_id=project_id,
            project_name=project_name,
            paths=paths,
            metadata=metadata,
        )
        self._save_project(project_id, project_data)

    def get_project(self, project_name: str) -> ProjectMetadata:
        """Get project data from manifest.

        Args:
            project_name: Name of the project

        Returns:
            Project metadata
        """
        project_id = get_project_id(project_name)
        return self._load_project(project_id)

    def list_projects(self) -> Iterator[ProjectMetadata]:
        """List all projects in the manifest.

        Returns:
            Iterator of project metadata
        """
        projects_dir = self.storage.projects_dir
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                manifest_path = project_dir / "manifest.json"
                if manifest_path.exists():
                    project_id = project_dir.name
                    yield self._load_project(project_id)
