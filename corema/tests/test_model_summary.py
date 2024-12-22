"""Tests for model summary pipeline."""

import logging
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import json

from ..pipelines.model_summary import ModelSummaryPipeline, ModelDetails
from ..storage import LocalStorage
from ..config import get_config
from ..utils.openai_models import DEFAULT_MODEL
from .utils import check_openai_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def storage() -> Mock:
    """Fixture for mocked storage."""
    mock_storage = Mock(spec=LocalStorage)
    # Mock get_project_dir to return a temporary path
    mock_storage.get_project_dir.return_value = Path("/tmp/test_project")
    return mock_storage


@pytest.fixture
def pipeline(storage: Mock) -> ModelSummaryPipeline:
    """Fixture for model summary pipeline instance."""
    return ModelSummaryPipeline(storage)


@pytest.mark.llm
@pytest.mark.skipif(
    not check_openai_key(get_config().get("OPENAI_API_KEY")),
    reason="OPENAI_API_KEY not configured or invalid",
)
def test_process_paper_llm(storage: Mock, tmp_path: Path) -> None:
    """Test LLM-based model detail extraction with a sample paper excerpt."""
    # Get the real API key from the environment
    real_config = get_config()
    api_key = real_config.get("OPENAI_API_KEY")

    # Create a test paper file
    paper_text = """
    We introduce two variants of our foundation model: GPT-4 (1.76 trillion parameters)
    and GPT-4-Base (350 billion parameters). Both models were trained using a combination
    of supervised learning and reinforcement learning from human feedback (RLHF).

    Training Infrastructure:
    - Initial training used 25,000 NVIDIA A100-80GB GPUs
    - Fine-tuning phase used a mix of 10,000 A100s and 5,000 H100 GPUs
    - Training time: 6 months for base model, additional 3 months for larger variant
    - Estimated costs: $100M for base model, $250M total
    - Primary training on Azure's AI supercomputing infrastructure
    - Some experiments run on internal HPC clusters and AWS P4d instances
    - Implemented DeepSpeed ZeRO-3 and Megatron-LM for distributed training
    - Used combination of data, tensor, and pipeline parallelism

    Architecture:
    - Both use decoder-only transformer architecture
    - Sparse mixture of experts layers in larger variant
    - Dense transformer blocks in base model
    - Custom attention mechanisms for long sequences
    - Built using PyTorch, JAX, and CUDA
    - Integrated with ONNX Runtime and TensorRT
    - Experimented with both dense and MoE variants

    Dataset:
    - 1.5 petabytes of filtered web text
    - 100B tokens from academic papers
    - Additional 50B tokens from code repositories
    - Multimodal data including images, code, and structured data
    - Preprocessing pipeline using Apache Beam and custom tools
    - Data filtering with multiple ML-based classifiers

    Evaluation:
    - Comprehensive testing on SuperGLUE, MMLU, HumanEval, and HELM
    - Achieved SOTA on 89% of benchmarks
    - Extensive red-teaming for safety
    - Additional domain-specific evaluations in science and medicine
    """

    paper_path = tmp_path / "test_paper.txt"
    paper_path.write_text(paper_text)

    # Mock manifest data
    mock_manifest_data = {
        "project_id": "test_project",
        "paths": {
            "papers": [
                {
                    "url": "http://example.com/paper",
                    "path": str(paper_path.with_suffix("")),
                }
            ]
        },
    }

    # Set up project results directory
    project_results_dir = (
        tmp_path / "projects" / "test_project" / "results" / "model_summary"
    )
    project_results_dir.mkdir(parents=True, exist_ok=True)

    # Update storage mock to use tmp_path
    storage.get_project_dir.return_value = tmp_path / "projects" / "test_project"

    # Configure with real OpenAI API key and mock manifest
    with patch("corema.config.get_config") as mock_get_config, patch(
        "corema.utils.manifest.ManifestManager.get_project"
    ) as mock_get_project:

        mock_config = {
            "pipeline": {
                "model_summary": {
                    "openai": {
                        "model": DEFAULT_MODEL,
                        "temperature": 0.0,
                    },
                },
            },
            "OPENAI_API_KEY": api_key,
        }
        mock_get_config.return_value = mock_config
        mock_get_project.return_value = mock_manifest_data

        # Create pipeline with mocked config
        pipeline = ModelSummaryPipeline(storage)
        pipeline.config = mock_config  # Override the config directly

        # Process the project
        results = pipeline.process_project("test_project")

        # Verify results were saved to project directory
        project_results_path = project_results_dir / f"{pipeline.PIPELINE_NAME}.jsonl"
        assert project_results_path.exists(), "Project results file should exist"

        # Read and verify saved results
        with open(project_results_path) as f:
            saved_results = [json.loads(line) for line in f]
            assert len(saved_results) > 0, "Should have saved at least one result"
            assert "project_id" in saved_results[0], "Results should include project_id"
            assert (
                saved_results[0]["project_id"] == "test_project"
            ), "Project ID should match"
            assert "paper_path" in saved_results[0], "Results should include paper_path"

        # Verify the extracted information
        assert len(results) > 0, "Should have processed at least one paper"
        result = next(iter(results.values()))
        assert isinstance(result, ModelDetails)

        # Check compute resources
        assert (
            len(result.compute_resources["gpu_count"]) > 0
        ), "Should extract some GPU count"
        assert (
            len(result.compute_resources["gpu_type"]) > 0
        ), "Should extract some GPU type"

        # Check architecture
        assert (
            len(result.architecture["model_type"]) > 0
        ), "Should extract some model type"
        assert (
            len(result.architecture["parameter_count"]) > 0
        ), "Should extract some parameter count"
        assert (
            len(result.architecture["key_components"]) > 0
        ), "Should extract some architecture components"

        # Check training stack
        assert (
            len(result.training_stack["frameworks"]) > 0
        ), "Should extract some frameworks"

        # Check dataset
        assert len(result.dataset["size"]) > 0, "Should extract dataset size"
        assert len(result.dataset["modalities"]) > 0, "Should extract data modalities"

        # Check evaluation
        assert (
            len(result.evaluation["datasets"]) > 0
        ), "Should extract evaluation datasets"
