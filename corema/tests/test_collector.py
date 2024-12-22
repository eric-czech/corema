# type: ignore
import pytest
from typing import Dict, Any, List

from ..collector import DataCollector
from ..storage import LocalStorage
from ..utils.names import get_project_id


@pytest.fixture(autouse=True)
def setup_storage(monkeypatch, tmp_path):
    """Set up storage configuration for tests."""
    monkeypatch.setenv("COLLECTOR_CONFIG_PATH", str(tmp_path / "config.yaml"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
storage:
    base_path: {str(tmp_path)}
"""
    )


@pytest.fixture
def collector(tmp_path):
    """Create a DataCollector instance with local storage."""
    storage = LocalStorage(str(tmp_path))
    return DataCollector(storage)


def test_process_project_success(collector, tmp_path):
    """Test successful processing of all URL types."""

    def mock_process_urls(
        project_name: str, urls: List[str], _
    ) -> List[Dict[str, Any]]:
        return [
            {
                "success": True,
                "url": url,
                "path": f"/path/to/{url.split('/')[-1]}",
                "github_urls": (
                    ["https://github.com/user/repo1"] if "paper" in url else []
                ),
                "doc_urls": (
                    ["https://docs.example.com/doc1"] if "github" in url else []
                ),
            }
            for url in urls
        ]

    collector._process_urls = mock_process_urls

    # Test data with all URL types
    project_name = "test-project"
    project_id = get_project_id(project_name)
    project_data = {
        "project_name": project_name,
        "paper_urls": ["https://example.com/paper1", "https://example.com/paper2"],
        "github_urls": ["https://github.com/user/repo1"],
        "doc_urls": ["https://docs.example.com/doc1"],
        "article_urls": ["https://blog.example.com/article1"],
    }

    collector.process_project(project_data)

    # Check that paths were tracked
    manifest = collector.storage.load_manifest(project_id)
    assert manifest is not None
    assert len(manifest["paths"]["papers"]) == 2
    assert len(manifest["paths"]["github_repos"]) == 1
    assert len(manifest["paths"]["documentation"]) == 1
    assert len(manifest["paths"]["articles"]) == 1


def test_process_project_failures(collector, tmp_path):
    """Test handling of failed URL processing."""

    def mock_process_urls(
        project_name: str, urls: List[str], _
    ) -> List[Dict[str, Any]]:
        return [
            {
                "success": i % 2 == 0,  # Alternate success/failure
                "url": url,
                "path": f"/path/to/{url.split('/')[-1]}" if i % 2 == 0 else None,
                "github_urls": (
                    ["https://github.com/user/repo1"]
                    if i % 2 == 0 and "paper" in url
                    else []
                ),
                "doc_urls": (
                    ["https://docs.example.com/doc1"]
                    if i % 2 == 0 and "github" in url
                    else []
                ),
            }
            for i, url in enumerate(urls)
        ]

    collector._process_urls = mock_process_urls

    # Test data with multiple URLs
    project_name = "test-project"
    project_id = get_project_id(project_name)
    project_data = {
        "project_name": project_name,
        "paper_urls": ["https://example.com/paper1", "https://example.com/paper2"],
        "github_urls": [
            "https://github.com/user/repo1",
            "https://github.com/user/repo2",
        ],
        "doc_urls": ["https://docs.example.com/doc1", "https://docs.example.com/doc2"],
        "article_urls": [
            "https://blog.example.com/article1",
            "https://blog.example.com/article2",
        ],
    }

    collector.process_project(project_data)

    # Check that only successful URLs were tracked
    manifest = collector.storage.load_manifest(project_id)
    assert manifest is not None
    assert len(manifest["paths"]["papers"]) == 1
    assert len(manifest["paths"]["github_repos"]) == 1
    assert len(manifest["paths"]["documentation"]) == 1
    assert len(manifest["paths"]["articles"]) == 1


def test_process_project_inferred_urls(collector, tmp_path):
    """Test that URLs inferred from papers and GitHub repos are added to project data."""

    def mock_process_urls(
        project_name: str, urls: List[str], _
    ) -> List[Dict[str, Any]]:
        if any("paper" in url for url in urls):
            return [
                {
                    "success": True,
                    "url": url,
                    "path": f"/path/to/{url.split('/')[-1]}",
                    "github_urls": ["https://github.com/user/inferred1"],
                    "doc_urls": [],
                }
                for url in urls
            ]
        elif any("github" in url for url in urls):
            return [
                {
                    "success": True,
                    "url": url,
                    "path": f"/path/to/{url.split('/')[-1]}",
                    "github_urls": [],
                    "doc_urls": ["https://docs.example.com/inferred1"],
                }
                for url in urls
            ]
        return [
            {
                "success": True,
                "url": url,
                "path": f"/path/to/{url.split('/')[-1]}",
                "github_urls": [],
                "doc_urls": [],
            }
            for url in urls
        ]

    collector._process_urls = mock_process_urls

    # Test data with initial URLs
    project_name = "test-project"
    project_id = get_project_id(project_name)
    project_data = {
        "project_name": project_name,
        "paper_urls": ["https://example.com/paper1"],
        "github_urls": ["https://github.com/user/repo1"],
    }

    collector.process_project(project_data)

    # Check that inferred URLs were added
    manifest = collector.storage.load_manifest(project_id)
    assert manifest is not None
    metadata = manifest["metadata"]
    assert "https://github.com/user/inferred1" in metadata["github_urls"]
    assert "https://docs.example.com/inferred1" in metadata["doc_urls"]
