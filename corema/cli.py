import fire
from pathlib import Path
import yaml
import logging
import shutil
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
import pandas as pd

from .collector import DataCollector
from .storage import LocalStorage
from .utils.names import get_project_id
from .config import get_config
from .pipelines.model_summary import ModelSummaryPipeline
from .utils.manifest import ManifestManager

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class CollectorCLI:
    def __init__(self) -> None:
        self.storage = LocalStorage()
        self.config = get_config()
        self.manifest = ManifestManager()
        self.default_output_dir = self.config["storage"]["base_path"]

    def _normalize_projects(self, projects: str | tuple) -> list[str]:
        """
        Convert a comma-separated string or tuple of project names or IDs into a list of normalized project IDs.

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

    def reset(self, output_dir: str | None = None) -> None:
        """
        Remove all collected data.

        Args:
            output_dir: Directory containing collected data. If None, uses the path from config.
        """
        output_path = Path(
            output_dir if output_dir is not None else self.default_output_dir
        )
        if output_path.exists():
            shutil.rmtree(output_path)
            logging.info(f"Removed all data in {output_path!r}")

    def collect(
        self,
        input_file: str,
        output_dir: str | None = None,
        max_workers: int = 1,
        overwrite: bool = False,
        projects: str = "",
    ) -> None:
        """
        Collect data for research model projects from a YAML file.

        Args:
            input_file: Path to YAML file containing project data
            output_dir: Directory to store collected data. If None, uses the path from config.
            max_workers: Number of worker threads for parallel project processing. Set to 1 for synchronous processing
            overwrite: If True, removes existing data for each project before processing
            projects: Comma-separated list of project names or IDs to process. If empty, process all projects.
        """
        output_path = Path(
            output_dir if output_dir is not None else self.default_output_dir
        )
        output_path.mkdir(parents=True, exist_ok=True)

        collector = DataCollector(storage=self.storage)

        # Parse and convert project names to IDs if provided
        project_id_list = self._normalize_projects(projects)

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
                    project_dir = self.storage.get_project_dir(
                        project_data["project_name"]
                    )
                    if project_dir.exists():
                        shutil.rmtree(project_dir)
                        logging.info(
                            f"Removed existing data for project {project_data['project_name']}"
                        )

            # Process projects with progress bar
            if max_workers == 1:
                # Single-threaded processing with progress bar
                for project_data in tqdm(
                    projects_to_process, desc="Processing projects"
                ):
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

    def summarize_models(
        self,
        output_file: str | None = None,
        projects: str = "",
        overwrite: bool = False,
        log_level: str = "INFO",
    ) -> None:
        """
        Process all papers and extract model details, saving results to both JSONL and Parquet files.
        Will load existing results for projects that aren't being processed and combine them with new results.

        Args:
            output_file: Base path for output files. If None, uses 'model_summaries' in the default output directory.
                Will append appropriate extensions (.jsonl and .parquet).
            projects: Comma-separated list of project names or IDs to process. If empty, processes all.
            overwrite: Whether to overwrite existing results for projects. If False, skips projects that have already been processed.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        setup_logging(log_level)

        # Initialize pipeline
        pipeline = ModelSummaryPipeline(self.storage)

        # Get list of projects to process
        if projects:
            projects_to_process = self._normalize_projects(projects)
        else:
            try:
                projects_to_process = [
                    p["project_id"] for p in self.manifest.list_projects()
                ]
            except Exception as e:
                logger.error(f"Error listing projects: {e}")
                return

        if not projects_to_process:
            logger.warning("No projects found to process")
            return

        # Get list of all projects for final aggregation
        try:
            all_projects = [p["project_id"] for p in self.manifest.list_projects()]
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
        results_dir = Path(self.default_output_dir) / "results" / pipeline.PIPELINE_NAME
        results_dir.mkdir(parents=True, exist_ok=True)

        # If output_file is provided, use it as the base name in the results directory
        base_name = Path(output_file).name if output_file else "model_summaries"
        base_path = results_dir / base_name

        # Save as JSONL
        jsonl_path = base_path.with_suffix(".jsonl")
        df.to_json(jsonl_path, orient="records", lines=True)
        logger.info(f"Saved combined model summaries to {jsonl_path}")


def main() -> None:
    fire.Fire(CollectorCLI)


if __name__ == "__main__":
    main()
