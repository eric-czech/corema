import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable, Any
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml

from corema.config import get_config
from corema.decorators import pipeline_task
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
from corema.utils.names import get_project_id

logger = logging.getLogger(__name__)


def get_publication_date(
    project_id: str, projects_data: Dict[str, Any]
) -> pd.Timestamp | None:
    """Get publication date from projects.yaml, paper metadata, or preprint URLs."""
    dates = []
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
                    dates.append(date)
                else:
                    date = pd.to_datetime(date_str)
                    logger.debug(f"  Found date in projects.yaml: {date}")
                    dates.append(date)
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
                                dates.append(date)
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
                            dates.append(date)

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
                            dates.append(date)
            except Exception as e:
                logger.debug(f"    Error parsing URL: {e}")

    # Filter out None values and get earliest date
    valid_dates = [d for d in dates if pd.notna(d)]
    if valid_dates:
        earliest_date = min(valid_dates)
        logger.debug(f"  Using earliest date: {earliest_date}")
        return earliest_date

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


def load_model_summary_dataset() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load and prepare the model summary dataset for visualizations."""
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])
    results_path = base_path / "results" / "model_summary" / "model_summaries.jsonl"

    # Read the model summaries
    if not results_path.exists():
        raise FileNotFoundError(f"Model summaries file not found at {results_path}")

    model_summaries = pd.read_json(results_path, lines=True)

    # Read projects.yaml
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Add publication dates and project names
    model_summaries["publication_date"] = model_summaries["project_id"].apply(
        lambda x: get_publication_date(x, projects)
    )
    model_summaries["project_name"] = model_summaries["project_id"].apply(
        lambda x: projects[x]["project_name"] if x in projects else x
    )

    # Get project fields
    project_fields = get_project_fields(model_summaries)

    # Sort by publication date
    model_summaries = model_summaries.sort_values("publication_date")

    return model_summaries, project_fields


def load_paper_metadata() -> pd.DataFrame:
    """Load paper metadata from all projects.

    Returns:
        pd.DataFrame: DataFrame containing paper metadata with columns:
            - project_id: str
            - project_name: str
            - paper_url: str
            - title: str
            - doi: str
            - publication_date: datetime
            - journal: str
            - authors: List[Dict] with keys 'name' and 'affiliations'
    """
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])

    # Read projects.yaml for project names
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Collect all paper metadata
    metadata_list = []
    for project_id, project in projects.items():
        paper_dir = base_path / "projects" / project_id / "paper"
        if not paper_dir.exists():
            continue

        # Find all JSON files in paper directory
        json_files = list(paper_dir.glob("*.json"))
        if not json_files:
            continue

        # Read each JSON file
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    metadata = json.load(f)

                # Add project info
                metadata["project_id"] = project_id
                metadata["project_name"] = project["project_name"]
                metadata["paper_url"] = json_file.stem  # URL hash is filename

                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(metadata_list)

    # Convert dates to datetime
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"])

    # Sort by publication date
    if "publication_date" in df.columns:
        df = df.sort_values("publication_date")

    return df


@pipeline_task
def visualize_model_summaries(
    allowlist: Optional[List[str]] = None,
    blocklist: Optional[List[str]] = None,
    images_dir: Path = Path("docs") / "images",
) -> None:
    """Generate all model summary visualizations.

    Args:
        allowlist: Optional list of visualizations to create. If provided, only these will be created.
        blocklist: Optional list of visualizations to skip. Only used if allowlist is None.
        images_dir: Directory where visualization images will be saved. Defaults to docs/images.

    Inputs:
        - data/results/model_summary/model_summaries.jsonl
        - resources/affiliations.csv
    Outputs:
        - {images_dir}/model_table.svg
        - {images_dir}/model_timeline.svg
        - {images_dir}/preprocessing_wordclouds.svg
        - {images_dir}/dataset_wordclouds.svg
        - {images_dir}/scientific_fields_wordcloud.svg
        - {images_dir}/training_details_wordclouds.svg
        - {images_dir}/training_platform_wordclouds.svg
        - {images_dir}/architecture_wordclouds.svg
        - {images_dir}/compute_resources_wordclouds.svg
        - {images_dir}/affiliations.svg
    Dependencies:
        - analyze_models
    """
    # Create output directory
    images_dir.mkdir(parents=True, exist_ok=True)

    # Set style and backend for SVG output
    plt.style.use("default")
    plt.switch_backend("svg")

    # Load datasets
    model_summaries, project_fields = load_model_summary_dataset()
    paper_metadata = load_paper_metadata()

    logger.info(f"Loaded {len(model_summaries)} models")
    logger.info(
        f"Models with publication dates: {model_summaries['publication_date'].notna().sum()}"
    )
    logger.info(f"Loaded {len(paper_metadata)} papers")

    # Define available visualizations with type hints
    all_viz: Dict[str, Tuple[Callable[..., None], List[Any]]] = {
        "model_table": (create_model_table, [model_summaries, images_dir]),
        "timeline": (
            create_timeline_visualization,
            [model_summaries, project_fields, images_dir],
        ),
        "preprocessing": (
            create_preprocessing_wordclouds,
            [model_summaries, images_dir],
        ),
        "dataset": (create_dataset_wordclouds, [model_summaries, images_dir]),
        "scientific_fields": (
            create_scientific_fields_wordcloud,
            [model_summaries, images_dir],
        ),
        "training_details": (
            create_training_details_wordclouds,
            [model_summaries, images_dir],
        ),
        "training_platform": (
            create_training_platform_wordclouds,
            [model_summaries, images_dir],
        ),
        "architecture": (create_architecture_wordclouds, [model_summaries, images_dir]),
        "compute_resources": (
            create_compute_resources_wordclouds,
            [model_summaries, images_dir],
        ),
        "affiliations": (
            create_affiliation_visualization,
            [paper_metadata, images_dir],
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
        logger.info(f"- Saved to '{images_dir / f'{name}.svg'}'")
