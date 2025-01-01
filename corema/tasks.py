"""Tasks for the corema package.

This module contains the main pipeline tasks for the corema package.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from corema.collector import DataCollector, ManifestManager
from corema.config import get_config
from corema.decorators import pipeline_task
from corema.pipelines.model_summary.extraction import ModelSummaryPipeline
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


@pipeline_task
def analyze_models(
    output_file: Optional[str] = None,
    projects: str = "",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Process all papers and extract model details, saving results to JSONL.

    Inputs:
        - data/results/model_summary/model_summaries.jsonl
    Outputs:
        - data/results/model_summary/analysis/
    Dependencies:
        - collect_data
    """
    # Initialize storage and config
    storage = LocalStorage()
    config = get_config()

    # Initialize pipeline
    pipeline = ModelSummaryPipeline(storage)

    # Get list of projects to process
    if projects:
        projects_to_process = _normalize_projects(projects)
    else:
        manifest = ManifestManager(base_path=str(storage.base_path))
        try:
            projects_to_process = [p["project_id"] for p in manifest.list_projects()]
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return

    if not projects_to_process:
        logger.warning("No projects found to process")
        return

    # Get list of all projects for final aggregation
    try:
        all_projects = [p["project_id"] for p in manifest.list_projects()]
    except Exception as e:
        logger.error(f"Error listing all projects: {e}")
        return

    # Process each requested project
    for project_id in tqdm(projects_to_process, desc="Processing projects"):
        try:
            logger.info(f"Processing project {project_id}")
            if not overwrite and pipeline.has_been_processed(project_id):
                logger.info(
                    f"Skipping project {project_id} - already processed and overwrite=False"
                )
                continue

            # Delete existing pipeline data if it exists
            results_dir = pipeline.get_project_results_dir(project_id)
            if results_dir.exists():
                shutil.rmtree(results_dir)
                logger.info(f"Removed existing model summary data at {results_dir}")

            pipeline.process_project(project_id)
        except Exception as e:
            logger.error(f"Error processing project {project_id}: {e}")

    # Load existing results for all projects
    all_results = []
    for project_id in all_projects:
        try:
            results = pipeline.load_results(project_id)
            for paper_path, details in results.items():
                record = details.model_dump()
                record["project_id"] = project_id
                record["paper_path"] = paper_path
                all_results.append(record)
        except Exception as e:
            logger.error(
                f"Error loading existing results for project {project_id}: {e}"
            )

    if not all_results:
        logger.warning("No model details were extracted")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Determine output paths for combined results
    results_dir = (
        Path(config["storage"]["base_path"]) / "results" / pipeline.PIPELINE_NAME
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # If output_file is provided, use it as the base name in the results directory
    base_name = Path(output_file).name if output_file else "model_summaries"
    base_path = results_dir / base_name

    # Save as JSONL
    jsonl_path = base_path.with_suffix(".jsonl")
    df.to_json(jsonl_path, orient="records", lines=True)
    logger.info(f"Saved combined model summaries to {jsonl_path!r}")
