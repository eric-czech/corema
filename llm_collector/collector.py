from typing import Dict, Any
import dramatiq
from .storage import LocalStorage
from .utils.manifest import ManifestManager
from .processors.paper import PaperProcessor
from .processors.github import GitHubProcessor
from .processors.docs import DocsProcessor

class DataCollector:
    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.manifest = ManifestManager()
        self.paper_processor = PaperProcessor(storage)
        self.github_processor = GitHubProcessor(storage)
        self.docs_processor = DocsProcessor(storage)

    def process_project(self, project_data: Dict[str, Any]):
        """Process a single project's data collection."""
        project_name = project_data["project_name"]
        
        # Track paths for manifest
        project_paths = {
            "papers": [],
            "github_repos": [],
            "documentation": []
        }

        # Process papers
        for paper_url in project_data.get("paper_urls", []):
            paper_job = self.paper_processor.process_paper.send(paper_url)
            project_paths["papers"].append(str(self.storage.get_path_for_url(paper_url)))
            
            # Infer GitHub URLs from paper
            github_urls = self.paper_processor.infer_github_urls(paper_url)
            project_data.setdefault("github_urls", []).extend(github_urls)

        # Process GitHub repositories
        for github_url in project_data.get("github_urls", []):
            github_job = self.github_processor.process_repository.send(github_url)
            project_paths["github_repos"].append(str(self.storage.get_path_for_url(github_url)))
            
            # Infer documentation URLs from GitHub
            doc_urls = self.github_processor.infer_doc_urls(github_url)
            project_data.setdefault("doc_urls", []).extend(doc_urls)

        # Process documentation
        for doc_url in project_data.get("doc_urls", []):
            doc_job = self.docs_processor.process_documentation.send(doc_url)
            project_paths["documentation"].append(str(self.storage.get_path_for_url(doc_url)))

        # Update manifest
        self.manifest.add_project(project_name, {
            "paths": project_paths,
            "metadata": project_data
        }) 