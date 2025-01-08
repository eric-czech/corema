"""Common functionality shared across pipeline modules."""

import logging
import shutil
from pathlib import Path
import json
import yaml
from typing import Dict, Any, TypeVar, Generic, List
from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm

from corema.config import get_config, get_openai_params
from corema.utils.manifest import ManifestManager
from corema.storage import LocalStorage
from corema.utils.names import get_project_id

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PipelineBase(Generic[T]):
    """Base class for pipeline implementations."""

    def __init__(self, pipeline_name: str, storage: LocalStorage):
        """Initialize pipeline.

        Args:
            pipeline_name: Name of the pipeline
            storage: Storage backend
        """
        self.pipeline_name = pipeline_name
        self.storage = storage
        self.config = get_config()
        self.manifest = ManifestManager()

    def get_project_results_dir(self, project_id: str) -> Path:
        """Get the directory for storing project-specific results."""
        return self.storage.get_project_dir(project_id) / "results" / self.pipeline_name

    def get_project_results_path(self, project_id: str) -> Path:
        """Get the path for storing project results."""
        return self.get_project_results_dir(project_id) / f"{self.pipeline_name}.json"

    def has_been_processed(self, project_id: str) -> bool:
        """Check if a project has already been processed."""
        results_path = self.get_project_results_path(project_id)
        return results_path.exists()

    def save_project_results(self, project_id: str, results: Dict[str, T]) -> None:
        """Save project results to disk."""
        results_path = self.get_project_results_path(project_id)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results for project {project_id!r} to {results_path!r}")
        logger.debug(f"Results to save: {results}")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.debug("Successfully wrote results to file")

    def process_paper(self, project_id: str, project_name: str, paper_path: Path) -> T:
        """Process a single paper.

        Args:
            project_id: ID of the project
            project_name: Name of the project (used in prompts)
            paper_path: Path to the paper text file

        Returns:
            Processed results for the paper
        """
        raise NotImplementedError("Subclasses must implement process_paper")

    def process_project(self, project_id: str) -> Dict[str, T]:
        """Process all papers for a project.

        Args:
            project_id: ID of the project to process

        Returns:
            Dictionary mapping paper paths/URLs to their processed results
        """
        try:
            # Get project metadata from manifest
            project = self.manifest.get_project(project_id)
            project_name = project["metadata"]["project_name"]

            results = {}
            # Process each paper listed in the manifest
            for paper in project["paths"]["papers"]:
                # Add .txt suffix to the path
                paper_path = Path(paper["path"]).with_suffix(".txt")

                # Check if the text file exists
                if not paper_path.exists():
                    logger.warning(f"Text file not found for paper: {paper_path!r}")
                    continue

                try:
                    result = self.process_paper(project_id, project_name, paper_path)
                    # Ensure result is a Pydantic BaseModel
                    if not isinstance(result, BaseModel):
                        raise TypeError(
                            f"Expected Pydantic BaseModel, got {type(result)}"
                        )

                    # Convert to dict and add metadata
                    result_dict = result.model_dump()
                    result_dict["project_id"] = project_id
                    result_dict["paper_url"] = paper[
                        "url"
                    ]  # Will raise KeyError if missing
                    result_dict["paper_hash"] = paper_path.name
                    results[paper_path.name] = result_dict
                except Exception as e:
                    logger.error(f"Error processing paper {paper_path!r}: {e}")

            if not results:
                logger.warning(
                    f"No papers were successfully processed for project {project_id!r}"
                )
            else:
                # Save project results
                self.save_project_results(project_id, results)

            return results

        except Exception as e:
            logger.error(f"Error accessing project {project_id!r}: {e}")
            return {}

    def load_results(self, project_id: str) -> Dict[str, Any]:
        """Load results for a specific project.

        Args:
            project_id: ID of the project to load results for

        Returns:
            Dictionary of results for the project
        """
        results_path = self.get_project_results_path(project_id)
        if not results_path.exists():
            raise FileNotFoundError(f"No results found for project {project_id!r}")

        with open(results_path) as f:
            results: Dict[str, Any] = json.load(f)
            return results

    def process_projects(
        self,
        projects: str = "",
        overwrite: bool = False,
    ) -> List[str]:
        """Process multiple projects.

        Args:
            projects: Comma-separated list of project IDs to process. If empty, process all.
            overwrite: Whether to overwrite existing results

        Returns:
            List of successfully processed project IDs
        """
        # Get list of projects to process
        if projects:
            projects_to_process = _normalize_projects(projects)
        else:
            try:
                projects_to_process = [
                    p["project_id"] for p in self.manifest.list_projects()
                ]
            except Exception as e:
                logger.error(f"Error listing projects: {e}")
                return []

        if not projects_to_process:
            logger.warning("No projects found to process")
            return []

        processed_projects = []
        # Process each requested project
        for project_id in tqdm(projects_to_process, desc="Processing projects"):
            try:
                logger.info(f"Processing project {project_id}")
                if not overwrite and self.has_been_processed(project_id):
                    logger.info(
                        f"Skipping project {project_id} - already processed and overwrite=False"
                    )
                    continue

                # Delete existing pipeline data if it exists
                results_dir = self.get_project_results_dir(project_id)
                if results_dir.exists():
                    shutil.rmtree(results_dir)
                    logger.info(f"Removed existing data at {results_dir}")

                self.process_project(project_id)
                processed_projects.append(project_id)
            except Exception as e:
                logger.error(f"Error processing project {project_id}: {e}")

        return processed_projects

    def consolidate_results(self) -> pd.DataFrame:
        """Consolidate all project results into a single DataFrame and save to JSONL.

        Returns:
            DataFrame of consolidated results
        """
        logger.info("Consolidating pipeline results")

        all_results = []
        for project_metadata in self.manifest.list_projects():
            project_id = project_metadata["project_id"]

            if not self.has_been_processed(project_id):
                logger.warning(f"No results found for project {project_id!r}")
                continue

            try:
                results = self.load_results(project_id)
                for paper_hash, result in results.items():
                    record = result
                    record["project_id"] = project_id
                    record["paper_hash"] = paper_hash
                    all_results.append(record)
            except Exception as e:
                logger.error(f"Error loading results for {project_id}: {e}")

        if not all_results:
            logger.warning("No results were found to consolidate")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Save as JSONL
        results_dir = (
            Path(self.config["storage"]["base_path"]) / "results" / self.pipeline_name
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Use pipeline name for output file
        jsonl_path = results_dir / f"{self.pipeline_name}.json"
        df.to_json(jsonl_path, orient="records", lines=True)
        logger.info(f"Saved consolidated results to {jsonl_path!r}")

        return df

    def _flatten_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Override this method to implement custom flattening logic."""
        return [results]


def load_projects_data() -> Dict[str, Any]:
    """Load and parse projects.yaml file.

    Returns:
        Dictionary mapping project IDs to project data
    """
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        return {get_project_id(p["project_name"]): p for p in projects_list}


def check_duplicate_project_ids(df: Any, id_column: str = "project_id") -> None:
    """Check for and log any duplicate project IDs in a DataFrame.

    Args:
        df: DataFrame to check
        id_column: Name of the column containing project IDs
    """
    duplicates = df[df.duplicated(subset=[id_column], keep=False)]
    if not duplicates.empty:
        logger.warning(
            f"Found {len(duplicates)} duplicate entries for project IDs: "
            f"{duplicates[id_column].unique().tolist()!r}"
        )


def get_openai_pipeline_params(pipeline_name: str) -> Dict[str, Any]:
    """Get OpenAI parameters for a specific pipeline from config.

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        Dictionary of OpenAI parameters
    """
    config = get_config()
    return get_openai_params(
        config.get("pipeline", {}).get(pipeline_name, {}).get("openai", {})
    )


def _normalize_projects(projects: str | tuple) -> List[str]:
    """Normalize project list from string or tuple to list of project IDs.

    Args:
        projects: Comma-separated string or tuple of project IDs or names

    Returns:
        List of normalized project IDs
    """
    if not projects:
        return []
    if isinstance(projects, str):
        project_list = projects.split(",")
    else:
        project_list = list(projects)
    return [get_project_id(p.strip()) for p in project_list]
