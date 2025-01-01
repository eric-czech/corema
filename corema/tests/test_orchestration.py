"""Tests for the orchestration system."""

import pytest

from corema.decorators import pipeline_task


@pipeline_task
def test_task() -> None:
    """A test task.

    Inputs:
        - data/input.txt
    Outputs:
        - data/output.txt
    Dependencies:
        - other_task
    """
    pass


def test_pipeline_task_registration() -> None:
    """Test that tasks are properly registered with their metadata."""
    from corema.decorators import PIPELINE_TASKS

    assert "test_task" in PIPELINE_TASKS
    metadata = PIPELINE_TASKS["test_task"]
    assert metadata["dependencies"] == ["other_task"]
    assert metadata["input_artifacts"] == ["data/input.txt"]
    assert metadata["output_artifacts"] == ["data/output.txt"]


def test_missing_metadata() -> None:
    """Test that missing metadata raises an error."""
    with pytest.raises(ValueError):

        @pipeline_task
        def no_metadata() -> None:
            """A task with no metadata."""
            pass


def test_multiple_tasks() -> None:
    """Test that multiple tasks can be registered."""
    from corema.decorators import PIPELINE_TASKS

    @pipeline_task
    def task1() -> None:
        """First task.

        Inputs:
            - data/input1.txt
        Outputs:
            - data/output1.txt
        Dependencies:
            - task2
        """
        pass

    @pipeline_task
    def task2() -> None:
        """Second task.

        Inputs:
            - data/input2.txt
        Outputs:
            - data/output2.txt
        Dependencies:
        """
        pass

    assert "task1" in PIPELINE_TASKS
    assert "task2" in PIPELINE_TASKS
    assert PIPELINE_TASKS["task1"]["dependencies"] == ["task2"]
    assert PIPELINE_TASKS["task2"]["dependencies"] == []


def test_get_task_dependencies() -> None:
    """Test getting task dependencies including transitive ones."""
    from corema.orchestration import PipelineOrchestrator

    @pipeline_task
    def task_a() -> None:
        """Task A.

        Inputs:
            - data/a.txt
        Outputs:
            - data/a_out.txt
        Dependencies:
            - task_b
        """
        pass

    @pipeline_task
    def task_b() -> None:
        """Task B.

        Inputs:
            - data/b.txt
        Outputs:
            - data/b_out.txt
        Dependencies:
            - task_c
        """
        pass

    @pipeline_task
    def task_c() -> None:
        """Task C.

        Inputs:
            - data/c.txt
        Outputs:
            - data/c_out.txt
        Dependencies:
        """
        pass

    orchestrator = PipelineOrchestrator()
    deps = orchestrator.get_task_dependencies("task_a")
    assert deps == {"task_b", "task_c"}
