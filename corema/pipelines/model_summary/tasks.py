from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable, Any
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

logger = logging.getLogger(__name__)


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
    results_path = (
        Path(get_config()["storage"]["base_path"])
        / "results"
        / "model_summary"
        / "model_summaries.jsonl"
    )
    model_summaries = pd.read_json(results_path, lines=True)

    # Get project fields
    project_fields = {}
    for idx, row in model_summaries.iterrows():
        fields = row["scientific_fields"]
        if isinstance(fields, (list, np.ndarray)):
            fields = [str(f).lower().strip() for f in fields if f is not None]
            project_fields[row["project_id"]] = fields
        else:
            project_fields[row["project_id"]] = []

    # Define available visualizations with type hints
    VisualizationFunc = Callable[..., None]
    all_viz: Dict[str, Tuple[VisualizationFunc, List[Any]]] = {
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
            [model_summaries, images_dir],
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
