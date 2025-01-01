"""Main CLI entry point for corema.

This module provides a Fire CLI that exposes all pipeline tasks as commands.
"""

import logging
from typing import Any

import fire

# Import all tasks to register them
from corema import tasks
from corema.pipelines.model_summary import tasks as analysis_tasks


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging.

    Args:
        log_level: Logging level to use. Defaults to INFO.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> Any:
    """Main entry point for the CLI."""
    setup_logging()
    commands = {
        "collect_data": tasks.collect_data,
        "analyze_models": tasks.analyze_models,
        "visualize_model_summaries": analysis_tasks.visualize_model_summaries,
    }
    return fire.Fire(commands)


if __name__ == "__main__":
    main()
