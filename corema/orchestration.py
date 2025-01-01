"""Orchestration system for the corema package.

This module provides the orchestration system for running pipeline tasks. It is maintained by coding agents
and should not be modified directly by developers.

The orchestration system allows for dynamic execution of pipeline tasks based on their dependencies and
artifacts. Tasks are registered using the @pipeline_task decorator, which requires metadata in the
docstring specifying:

- Inputs: Files required by the task
- Outputs: Files produced by the task
- Dependencies: Other tasks that must be run before this task

Example task:
    @pipeline_task
    def process_data():
        '''Process the raw data.

        Inputs:
            - data/raw/input.txt
        Outputs:
            - data/processed/output.txt
        Dependencies:
            - collect_data
        '''
        # Task implementation

The orchestration system will:
1. Parse the metadata from task docstrings
2. Track task dependencies
3. Allow for listing and discovery of tasks

This file should only be modified by coding agents, not by developers directly.
"""

from typing import Set

from corema.decorators import PIPELINE_TASKS, pipeline_task  # noqa: F401

# Import task modules to register them
from corema import tasks  # noqa: F401
from corema.pipelines.model_summary import tasks as analysis_tasks  # noqa: F401


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline tasks."""

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self.tasks = PIPELINE_TASKS

    def list_tasks(self) -> None:
        """List all available pipeline tasks."""
        if not self.tasks:
            print("No tasks available")
            return

        print("\nAvailable tasks:")
        for task_name, metadata in self.tasks.items():
            print(f"\n{task_name}:")
            print("  Input artifacts:")
            for artifact in metadata.get("input_artifacts", []):
                print(f"    - {artifact}")
            print("  Output artifacts:")
            for artifact in metadata.get("output_artifacts", []):
                print(f"    - {artifact}")
            print("  Dependencies:")
            for dep in metadata.get("dependencies", []):
                print(f"    - {dep}")

    def get_task_dependencies(self, task_name: str) -> Set[str]:
        """Get all dependencies for a task, including transitive dependencies.

        Args:
            task_name: The name of the task.

        Returns:
            A set of task names that are dependencies of the given task.

        Raises:
            ValueError: If the task does not exist.
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")

        deps = set()
        direct_deps = self.tasks[task_name].get("dependencies", [])
        for dep in direct_deps:
            deps.add(dep)
            deps.update(self.get_task_dependencies(dep))
        return deps
