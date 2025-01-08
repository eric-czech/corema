"""Main CLI entry point for corema.

This module provides a Fire CLI that exposes all pipeline tasks as commands.
"""

import logging
from typing import Any, Dict, Callable

import fire

from corema import tasks
from corema.pipelines.model_summary.tasks import (
    analyze_models,
    visualize_model_summaries,
)
from corema.pipelines.inductive_bias.tasks import analyze_inductive_biases

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> Any:
    """Main entry point for the CLI."""
    setup_logging()
    commands: Dict[str, Callable] = {
        "collect_data": tasks.collect_data,
        "analyze_models": analyze_models,
        "visualize_model_summaries": visualize_model_summaries,
        "analyze_inductive_biases": analyze_inductive_biases,
    }
    return fire.Fire(commands)


if __name__ == "__main__":
    main()
