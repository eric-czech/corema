"""Decorators for the corema package.

This module contains decorators used throughout the corema package. It is maintained by coding agents
and should not be modified directly by developers.
"""

import functools
import re
from typing import Any, Callable, Dict, TypedDict


class TaskMetadata(TypedDict):
    """Type definition for task metadata."""

    input_artifacts: list[str]
    output_artifacts: list[str]
    dependencies: list[str]


PIPELINE_TASKS: Dict[str, TaskMetadata] = {}


def parse_task_metadata(docstring: str) -> TaskMetadata:
    """Parse the metadata from a task's docstring.

    Args:
        docstring: The docstring to parse.

    Returns:
        A dictionary containing the parsed metadata.

    Raises:
        ValueError: If no metadata is found in the docstring.
    """
    if not docstring:
        raise ValueError("No docstring found")

    # Extract input artifacts
    input_match = re.search(r"Inputs:\s*\n((?:\s*-\s*[^\n]+\n)*)", docstring)
    input_artifacts = []
    if input_match:
        input_artifacts = [
            line.strip("- ").strip()
            for line in input_match.group(1).strip().split("\n")
            if line.strip()
        ]

    # Extract output artifacts
    output_match = re.search(r"Outputs:\s*\n((?:\s*-\s*[^\n]+\n)*)", docstring)
    output_artifacts = []
    if output_match:
        output_artifacts = [
            line.strip("- ").strip()
            for line in output_match.group(1).strip().split("\n")
            if line.strip()
        ]

    # Extract dependencies
    deps_match = re.search(r"Dependencies:\s*\n((?:\s*-\s*[^\n]+\n)*)", docstring)
    dependencies = []
    if deps_match:
        dependencies = [
            line.strip("- ").strip()
            for line in deps_match.group(1).strip().split("\n")
            if line.strip()
        ]

    if not input_artifacts and not output_artifacts and not dependencies:
        raise ValueError("No metadata found in docstring")

    return {
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "dependencies": dependencies,
    }


def pipeline_task(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a function as a pipeline task.

    The decorated function must have a docstring containing metadata in the following format:

    Inputs:
        - path/to/input1.txt
        - path/to/input2.txt
    Outputs:
        - path/to/output1.txt
        - path/to/output2.txt
    Dependencies:
        - other_task
        - another_task

    Example:
        @pipeline_task
        def my_task():
            '''Process some data.

            Inputs:
                - data/input.txt
            Outputs:
                - data/output.txt
            Dependencies:
                - other_task
            '''
            pass

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.

    Raises:
        ValueError: If the function has no docstring or no metadata.
    """
    metadata = parse_task_metadata(func.__doc__ or "")
    PIPELINE_TASKS[func.__name__] = metadata

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, "_pipeline_metadata", metadata)
    return wrapper
