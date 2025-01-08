"""Tasks for the corema package.

This module contains the main pipeline tasks for the corema package.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from corema.collector import DataCollector
from corema.config import get_config
from corema.decorators import pipeline_task
from corema.storage import LocalStorage
from corema.utils.names import get_project_id

logger = logging.getLogger(__name__)


def _normalize_projects(projects: str | tuple) -> list[str]:
    """Convert a comma-separated string or tuple of project names or IDs into a list of normalized project IDs.

    Args:
        projects: Comma-separated string or tuple of project names or IDs

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


@pipeline_task
def collect_data(
    input_file: str,
    output_dir: Optional[str] = None,
    max_workers: int = 1,
    overwrite: bool = False,
    projects: str = "",
    **kwargs: Any,
) -> None:
    """Collect data for research model projects from a YAML file.

    Inputs:
        - data/raw/manifest.json
    Outputs:
        - data/results/model_summary/model_summaries.jsonl
    Dependencies:
    """
    # Initialize storage and config
    storage = LocalStorage()
    config = get_config()
    output_path = Path(
        output_dir if output_dir is not None else config["storage"]["base_path"]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    collector = DataCollector(storage=storage)

    # Parse and convert project names to IDs if provided
    project_id_list = _normalize_projects(projects)

    with open(input_file, "r") as f:
        projects_data = yaml.safe_load(f)
        # Filter projects based on project_ids and processing status
        projects_to_process = []
        for project_data in projects_data:
            project_name = project_data["project_name"]
            project_id = get_project_id(project_name)
            if not project_id_list or project_id in project_id_list:
                if overwrite or not collector.has_been_processed(project_id):
                    projects_to_process.append(project_data)
                else:
                    logging.info(
                        f"Skipping project {project_name} - already processed and overwrite=False"
                    )

        # Handle overwrite before processing
        if overwrite:
            for project_data in projects_to_process:
                project_dir = storage.get_project_dir(project_data["project_name"])
                if project_dir.exists():
                    shutil.rmtree(project_dir)
                    logging.info(
                        f"Removed existing data for project {project_data['project_name']}"
                    )

        # Process projects with progress bar
        if max_workers == 1:
            # Single-threaded processing with progress bar
            for project_data in tqdm(projects_to_process, desc="Processing projects"):
                collector.process_project(project_data)
        else:
            # Multi-threaded processing with progress bar
            def process_project(project_data: dict) -> None:
                collector.process_project(project_data)

            thread_map(
                process_project,
                projects_to_process,
                max_workers=max_workers,
                desc="Processing projects",
            )
