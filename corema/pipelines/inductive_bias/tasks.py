"""Tasks for analyzing inductive biases in scientific foundation models."""

import logging
from typing import Any

from corema.storage import LocalStorage
from corema.decorators import pipeline_task
from corema.pipelines.inductive_bias.extraction import InductiveBiasPipeline

logger = logging.getLogger(__name__)


@pipeline_task
def analyze_inductive_biases(
    projects: str = "",
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    """Extract inductive biases from papers and save results.

    Args:
        projects: Comma-separated list of project IDs to process. If empty, process all.
        overwrite: Whether to overwrite existing results.

    Inputs:
        - data/projects/*/papers/*.txt
    Outputs:
        - data/results/inductive_bias/inductive_bias.json
    Dependencies:
        - collect_data
    """
    # Initialize pipeline
    pipeline = InductiveBiasPipeline(LocalStorage())

    # Process projects
    pipeline.process_projects(projects=projects, overwrite=overwrite)

    # Consolidate and save results
    pipeline.consolidate_results()
