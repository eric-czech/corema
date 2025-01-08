"""Tasks for analyzing model details in scientific foundation models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from corema.config import get_config
from corema.decorators import pipeline_task
from corema.pipelines.model_summary.extraction import ModelSummaryPipeline
from corema.pipelines.model_summary.visualization import (
    create_model_table,
    create_timeline_visualization,
    create_preprocessing_wordclouds,
    create_dataset_wordclouds,
    create_scientific_fields_wordcloud,
    create_training_details_wordclouds,
    create_training_platform_wordclouds,
    create_architecture_wordclouds,
    create_compute_resources_wordclouds,
    create_affiliation_visualization,
)
from corema.storage import LocalStorage
from corema.utils.names import get_project_id

logger = logging.getLogger(__name__)


def get_publication_date(
    project_id: str, projects_data: Dict[str, Any]
) -> pd.Timestamp | None:
    """Get publication date from projects.yaml, paper metadata, or preprint URLs."""
    logger.debug(f"\nProcessing {project_id}:")

    # Try projects.yaml
    if project_id in projects_data:
        project = projects_data[project_id]
        if "publication_date" in project:
            try:
                date_str = project["publication_date"]
                # Handle YYYY-MM format
                if len(date_str.split("-")) == 2:
                    date = pd.to_datetime(date_str + "-01")
                    logger.debug(f"  Found date in projects.yaml: {date}")
                    return date
                else:
                    date = pd.to_datetime(date_str)
                    logger.debug(f"  Found date in projects.yaml: {date}")
                    return date
            except Exception as e:
                logger.debug(
                    f"  Could not parse date {date_str} from projects.yaml: {e}"
                )

    # Try paper metadata
    paper_dir = Path("data/projects") / project_id / "paper"
    if paper_dir.exists():
        json_files = list(paper_dir.glob("*.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    paper_data = json.load(f)
                    if "publication_date" in paper_data:
                        try:
                            date = pd.to_datetime(paper_data["publication_date"])
                            if pd.notna(date):
                                logger.debug(f"  Found date in paper metadata: {date}")
                                return date
                        except Exception as e:
                            logger.debug(
                                f"  Could not parse date from paper metadata: {e}"
                            )
            except Exception as e:
                logger.debug(f"  Error reading paper metadata: {e}")

    # Try extracting from preprint URLs in projects.yaml
    if project_id in projects_data and "paper_urls" in projects_data[project_id]:
        for url in projects_data[project_id]["paper_urls"]:
            logger.debug(f"  Checking URL: {url}")
            url = url.lower()

            try:
                # Handle arXiv URLs
                if "arxiv.org" in url:
                    arxiv_id = url.split("/")[-1].split("v")[0].replace(".pdf", "")
                    logger.debug(f"    arXiv ID: {arxiv_id}")

                    # Try YYMM format
                    if len(arxiv_id) >= 4 and arxiv_id[:4].isdigit():
                        year = "20" + arxiv_id[:2]
                        month = arxiv_id[2:4]
                        if 1 <= int(month) <= 12:
                            date = pd.to_datetime(f"{year}-{month}-01")
                            logger.debug(f"    Extracted date: {date}")
                            return date

                # Handle bioRxiv/medRxiv URLs
                elif any(x in url for x in ["biorxiv.org", "medrxiv.org"]):
                    if "10.1101/" in url:
                        date_str = url.split("10.1101/")[1].split(".")[0]
                        logger.debug(f"    bioRxiv/medRxiv date string: {date_str}")
                        if len(date_str) == 10:  # YYYY.MM.DD
                            year = date_str[:4]
                            month = date_str[5:7]
                            day = date_str[8:10]
                            date = pd.to_datetime(f"{year}-{month}-{day}")
                            logger.debug(f"    Extracted date: {date}")
                            return date
            except Exception as e:
                logger.debug(f"    Error parsing URL: {e}")

    logger.debug("  No date found")
    return None


def get_project_fields(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get scientific fields for each project."""
    project_fields = {}
    for idx, row in df.iterrows():
        fields = row["scientific_fields"]
        if isinstance(fields, (list, np.ndarray)):
            fields = [str(f).lower().strip() for f in fields if f is not None]
            project_fields[row["project_id"]] = fields
        else:
            project_fields[row["project_id"]] = []
    return project_fields


def load_model_summary_dataset() -> pd.DataFrame:
    """Load and prepare the model summary dataset for visualizations.

    Returns:
        DataFrame with model summaries
    """
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])
    results_path = base_path / "results" / "model_summary" / "model_summaries.jsonl"

    # Read the model summaries
    if not results_path.exists():
        raise FileNotFoundError(f"Model summaries file not found at {results_path}")

    model_summaries = pd.read_json(results_path, lines=True)

    # Read projects.yaml for project names
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Add project names
    model_summaries["project_name"] = model_summaries["project_id"].apply(
        lambda x: projects[x]["project_name"] if x in projects else x
    )

    # Check for duplicate project IDs
    duplicates = model_summaries[
        model_summaries.duplicated(subset=["project_id"], keep=False)
    ]
    if not duplicates.empty:
        logger.warning(
            f"Found {len(duplicates)} duplicate entries in model_summaries.jsonl for project IDs: "
            f"{duplicates['project_id'].unique().tolist()!r}"
        )

    return model_summaries


def load_project_metadata() -> pd.DataFrame:
    """Load project metadata from all projects, one entry per project.

    Builds metadata in layers:
    1. Start with projects.yaml data
    2. Add information from manifest.json files
    3. Add information from paper metadata files

    Returns:
        DataFrame with columns:
        - project_id: str
        - project_name: str
        - publication_date: datetime
        - paper_urls: List[str]
        - primary_paper_title: str
        - primary_paper_url: str
        - github_urls: List[str]
        - authors: List[Dict] with keys 'name' and 'affiliations'
    """
    # 1. Start with projects.yaml
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)

    # Create initial DataFrame
    df = pd.DataFrame(
        [
            {
                "project_id": get_project_id(p["project_name"]),
                "project_name": p["project_name"],
                "paper_urls": p.get("paper_urls", []),
                "github_urls": p.get("github_urls", []),
                "primary_paper_url": p.get("paper_urls", [None])[0],
            }
            for p in projects_list
        ]
    )

    # Check for duplicate project IDs
    duplicates = df[df.duplicated(subset=["project_id"], keep=False)]
    if not duplicates.empty:
        logger.warning(
            f"Found {len(duplicates)} duplicate entries in projects.yaml for project IDs: "
            f"{duplicates['project_id'].unique().tolist()!r}"
        )

    # 2. Add/update information from manifest files
    config = get_config()
    base_path = Path(config["storage"]["base_path"])
    projects_dir = base_path / "projects"

    for idx, row in df.iterrows():
        project_id = row["project_id"]
        manifest_path = projects_dir / project_id / "manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                # Get paper paths and URLs
                paper_paths = manifest.get("paths", {}).get("papers", [])
                if paper_paths:
                    # Store first paper's URL and path for metadata lookup
                    df.at[idx, "primary_paper_url"] = paper_paths[0]["url"]
                    df.at[idx, "primary_paper_path"] = paper_paths[0]["path"]

                # Update GitHub URLs from manifest
                github_urls = [
                    repo["url"]
                    for repo in manifest.get("paths", {}).get("github_repos", [])
                ]
                if github_urls:
                    df.at[idx, "github_urls"] = github_urls
            except Exception as e:
                logger.error(f"Error reading manifest for project {project_id}: {e}")

    # 3. Add paper metadata
    for idx, row in df.iterrows():
        paper_path = row.get("primary_paper_path")
        if pd.notna(paper_path):
            metadata_path = Path(str(paper_path) + ".json")
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        paper_metadata = json.load(f)
                        df.at[idx, "primary_paper_title"] = paper_metadata.get("title")
                        df.at[idx, "authors"] = paper_metadata.get("authors", [])
                except Exception as e:
                    logger.error(
                        f"Error reading paper metadata for {metadata_path}: {e}"
                    )

    # 4. Resolve publication dates using get_publication_date
    # Convert projects list to dict for get_publication_date
    projects = {get_project_id(p["project_name"]): p for p in projects_list}
    df["publication_date"] = df["project_id"].apply(
        lambda x: get_publication_date(x, projects)
    )

    # Clean up
    df = df.drop(columns=["primary_paper_path"], errors="ignore")

    # Clean up lists by removing null elements
    df["github_urls"] = df["github_urls"].apply(
        lambda x: [url for url in (x if isinstance(x, list) else []) if pd.notna(url)]
    )
    df["authors"] = df["authors"].apply(
        lambda x: [
            author for author in (x if isinstance(x, list) else []) if pd.notna(author)
        ]
    )

    # Sort by publication date
    df = df.sort_values("publication_date")

    return df


@pipeline_task
def visualize_model_summaries(
    allowlist: Optional[List[str]] = None,
    blocklist: Optional[List[str]] = None,
    results_dir: Path = Path("docs") / "results",
) -> None:
    """Generate all model summary visualizations.

    Args:
        allowlist: Optional list of visualizations to create. If provided, only these will be created.
        blocklist: Optional list of visualizations to skip. Only used if allowlist is None.
        results_dir: Directory where visualization results will be saved. Defaults to docs/results.

    Inputs:
        - data/results/model_summary/model_summaries.jsonl
        - resources/affiliations.csv
    Outputs:
        - {results_dir}/tables/model_table.md
        - {results_dir}/tables/model_table.html
        - {results_dir}/images/model_timeline.svg
        - {results_dir}/images/preprocessing_wordclouds.svg
        - {results_dir}/images/dataset_wordclouds.svg
        - {results_dir}/images/scientific_fields_wordcloud.svg
        - {results_dir}/images/training_details_wordclouds.svg
        - {results_dir}/images/training_platform_wordclouds.svg
        - {results_dir}/images/architecture_wordclouds.svg
        - {results_dir}/images/compute_resources_wordclouds.svg
        - {results_dir}/images/affiliations.svg
    Dependencies:
        - analyze_models
    """
    # Create output directories
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "images").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Set style and backend for SVG output
    plt.style.use("default")
    plt.switch_backend("svg")

    # Load datasets
    model_summaries = load_model_summary_dataset()
    project_metadata = load_project_metadata()

    # Add publication dates from project_metadata
    model_summaries = model_summaries.merge(
        project_metadata[["project_id", "publication_date"]],
        on="project_id",
        how="left",
    )
    model_summaries = model_summaries.sort_values("publication_date")

    # Get project fields
    project_fields = get_project_fields(model_summaries)

    logger.info(f"Loaded {len(model_summaries)} models")
    logger.info(
        f"Models with publication dates: {model_summaries['publication_date'].notna().sum()}"
    )
    logger.info(f"Loaded {len(project_metadata)} projects")

    # Define available visualizations with type hints
    all_viz: Dict[str, Tuple[Callable[..., None], List[Any]]] = {
        "model_table": (create_model_table, [project_metadata, results_dir]),
        "timeline": (
            create_timeline_visualization,
            [model_summaries, project_fields, results_dir],
        ),
        "preprocessing": (
            create_preprocessing_wordclouds,
            [model_summaries, results_dir],
        ),
        "dataset": (create_dataset_wordclouds, [model_summaries, results_dir]),
        "scientific_fields": (
            create_scientific_fields_wordcloud,
            [model_summaries, results_dir],
        ),
        "training_details": (
            create_training_details_wordclouds,
            [model_summaries, results_dir],
        ),
        "training_platform": (
            create_training_platform_wordclouds,
            [model_summaries, results_dir],
        ),
        "architecture": (
            create_architecture_wordclouds,
            [model_summaries, results_dir],
        ),
        "compute_resources": (
            create_compute_resources_wordclouds,
            [model_summaries, results_dir],
        ),
        "affiliations": (
            create_affiliation_visualization,
            [project_metadata, results_dir],
        ),
    }

    # Determine which visualizations to create
    if allowlist:
        invalid = set(allowlist) - set(all_viz.keys())
        if invalid:
            raise ValueError(f"Invalid visualization names in allowlist: {invalid!r}")
        viz_to_create = {k: v for k, v in all_viz.items() if k in allowlist}
    else:
        if blocklist:
            invalid = set(blocklist) - set(all_viz.keys())
            if invalid:
                raise ValueError(
                    f"Invalid visualization names in blocklist: {invalid!r}"
                )
            viz_to_create = {k: v for k, v in all_viz.items() if k not in blocklist}
        else:
            viz_to_create = all_viz

    # Create visualizations
    logger.info("\nGenerating visualizations...")
    for name, (func, args) in viz_to_create.items():
        logger.info(f"Creating {name}...")
        func(*args)
        if name == "model_table":
            logger.info(f"- Saved to '{results_dir / 'tables' / 'model_table.md'}'")
            logger.info(f"- Saved to '{results_dir / 'tables' / 'model_table.html'}'")
        else:
            logger.info(f"- Saved to '{results_dir / 'images' / f'{name}.svg'}'")


@pipeline_task
def analyze_models(
    projects: str = "",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Extract model details from papers and save results.

    Inputs:
        - data/projects/*/papers/*.txt
    Outputs:
        - data/results/model_summary/model_summary.json
    Dependencies:
        - collect_data
    """
    # Initialize pipeline
    pipeline = ModelSummaryPipeline(LocalStorage())

    # Process projects
    pipeline.process_projects(projects=projects, overwrite=overwrite)

    # Consolidate and save results
    pipeline.consolidate_results()
