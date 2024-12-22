import pytest
from pathlib import Path
import json

from ..utils.manifest import ManifestManager, PathMapping


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def manifest_manager(temp_data_dir: Path) -> ManifestManager:
    """Create a ManifestManager instance with a temporary data directory."""
    return ManifestManager(str(temp_data_dir))


def test_new_manifest(manifest_manager: ManifestManager) -> None:
    """Test creating a new manifest."""
    project = manifest_manager.get_project("test_project")
    assert project["project_id"] == "test_project"
    assert project["paths"] == {
        "papers": [],
        "github_repos": [],
        "documentation": [],
        "articles": [],
    }
    assert project["metadata"] == {}


def test_add_project_with_paths(
    manifest_manager: ManifestManager, temp_data_dir: Path
) -> None:
    """Test adding a project with paths."""
    paths: PathMapping = {
        "papers": [
            {"url": "https://example.com/paper.pdf", "path": "data/test/paper/123"}
        ],
        "github_repos": [
            {"url": "https://github.com/test/repo", "path": "data/test/github/456"}
        ],
        "documentation": [
            {"url": "https://docs.test.com", "path": "data/test/docs/789"}
        ],
        "articles": [],
    }

    project_data = {
        "paths": paths,
        "metadata": {
            "project_name": "Test Project",
            "paper_urls": ["https://example.com/paper.pdf"],
            "github_urls": ["https://github.com/test/repo"],
            "doc_urls": ["https://docs.test.com"],
        },
    }

    manifest_manager.add_project("test_project", project_data)

    # Verify manifest file was created in project directory
    manifest_file = temp_data_dir / "projects" / "test_project" / "manifest.json"
    assert manifest_file.exists()

    # Verify manifest content
    with open(manifest_file, "r") as f:
        stored_data = json.load(f)
        assert stored_data["project_id"] == "test_project"
        assert stored_data["paths"] == paths
        assert stored_data["metadata"] == project_data["metadata"]

    # Verify we can read it back
    project = manifest_manager.get_project("test_project")
    assert project["project_id"] == "test_project"
    assert len(project["paths"]["papers"]) == 1
    assert project["paths"]["papers"][0]["url"] == "https://example.com/paper.pdf"
    assert project["paths"]["papers"][0]["path"] == "data/test/paper/123"


def test_save_and_load_project(
    manifest_manager: ManifestManager, temp_data_dir: Path
) -> None:
    """Test saving and loading individual project files."""
    paths: PathMapping = {
        "papers": [
            {"url": "https://example.com/paper.pdf", "path": "data/test/paper/123"}
        ],
        "github_repos": [],
        "documentation": [],
        "articles": [],
    }

    project_data = {
        "paths": paths,
        "metadata": {
            "project_name": "Test Project",
            "paper_urls": ["https://example.com/paper.pdf"],
        },
    }

    manifest_manager.add_project("test_project", project_data)

    # Verify file was created in project directory
    manifest_file = temp_data_dir / "projects" / "test_project" / "manifest.json"
    assert manifest_file.exists()

    # Create new manager instance to test loading
    new_manager = ManifestManager(str(temp_data_dir))
    project = new_manager.get_project("test_project")

    assert project["project_id"] == "test_project"
    assert len(project["paths"]["papers"]) == 1
    assert project["paths"]["papers"][0]["url"] == "https://example.com/paper.pdf"
    assert project["paths"]["papers"][0]["path"] == "data/test/paper/123"


def test_list_projects(manifest_manager: ManifestManager, temp_data_dir: Path) -> None:
    """Test listing all projects in the manifest."""
    # Add a few test projects
    for i in range(3):
        project_data = {
            "paths": {
                "papers": [],
                "github_repos": [],
                "documentation": [],
                "articles": [],
            },
            "metadata": {"project_name": f"Test Project {i}"},
        }
        manifest_manager.add_project(f"test_project_{i}", project_data)

        # Verify manifest file was created
        manifest_file = (
            temp_data_dir / "projects" / f"test_project_{i}" / "manifest.json"
        )
        assert manifest_file.exists()

    # List all projects
    projects = list(manifest_manager.list_projects())
    assert len(projects) == 3
    assert all(p["project_id"].startswith("test_project_") for p in projects)
