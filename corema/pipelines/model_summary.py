from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging
from ..storage import LocalStorage
from ..utils.openai_api import get_structured_output
from ..config import get_config, get_openai_params
from ..resources.model_summary_prompts import (
    MODEL_SUMMARY_SYSTEM_PROMPT,
    MODEL_SUMMARY_USER_PROMPT,
)
from ..utils.manifest import ManifestManager
import json

logger = logging.getLogger(__name__)


class ModelDetails(BaseModel):
    """Structured information about a foundation model extracted from a paper."""

    scientific_fields: List[str] = Field(
        description="Most relevant fields of science this model is designed for or applicable to"
    )

    preprocessing: Dict[str, Any] = Field(
        description="Data preprocessing details including tools and systems used",
        default_factory=lambda: {
            "tools": [],  # List[str]
            "systems": [],  # List[str]
            "data_cleaning": [],  # List[str]
            "data_transformation": [],  # List[str]
            "data_augmentation": [],  # List[str]
            "normalization": [],  # List[str]
            "tokenization": [],  # List[str]
            "other_tools": [],  # List[str]
        },
    )

    compute_resources: Dict[str, Any] = Field(
        description="Computational resources used for training",
        default_factory=lambda: {
            "training_time": [],  # List[str]
            "cost_estimate": [],  # List[str]
            "gpu_count": [],  # List[int]
            "gpu_type": [],  # List[str]
            "other_hardware": [],  # List[str]
        },
    )

    training_platform: Dict[str, Any] = Field(
        description="Platforms and infrastructure used for training",
        default_factory=lambda: {
            "cloud_provider": [],  # List[str]
            "hpc_system": [],  # List[str]
            "training_service": [],  # List[str]
            "other_platforms": [],  # List[str]
        },
    )

    architecture: Dict[str, Any] = Field(
        description="Model architecture details",
        default_factory=lambda: {
            "model_type": [],  # List[str]
            "parameter_count": [],  # List[int]
            "architecture_type": [],  # List[str]
            "key_components": [],  # List[str]
        },
    )

    training_stack: Dict[str, Any] = Field(
        description="Software stack used for model development and training",
        default_factory=lambda: {
            "frameworks": [],  # List[str]
            "libraries": [],  # List[str]
            "languages": [],  # List[str]
            "tools": [],  # List[str]
        },
    )

    training_details: Dict[str, Any] = Field(
        description="Training methods and techniques used",
        default_factory=lambda: {
            "parallelization": [],  # List[str]
            "checkpointing": [],  # List[str]
            "optimization_methods": [],  # List[str]
            "regularization": [],  # List[str]
            "loss_functions": [],  # List[str]
            "training_techniques": [],  # List[str]
        },
    )

    dataset: Dict[str, Any] = Field(
        description="Training dataset details",
        default_factory=lambda: {
            "size": [],  # List[str]
            "modalities": [],  # List[str]
        },
    )

    evaluation: Dict[str, Any] = Field(
        description="Evaluation details",
        default_factory=lambda: {
            "datasets": [],  # List[str]
            "metrics": [],  # List[str]
            "key_results": {},  # Dict[str, List[Any]]
        },
    )

    use_cases: List[str] = Field(
        description="Intended downstream applications and use cases"
    )


def combine_model_details(a: ModelDetails, b: ModelDetails) -> ModelDetails:
    """Combine model details from two chunks, keeping the most complete information."""

    def merge_lists(list_a: List[Any], list_b: List[Any]) -> List[Any]:
        """Merge two lists with deduplication, preserving value types."""
        # Convert to strings for comparison but keep original values
        str_to_val = {}
        for val in list_a + list_b:
            str_val = str(val)
            # If we see the same string representation, keep the first value
            if str_val not in str_to_val:
                str_to_val[str_val] = val

        return list(str_to_val.values())

    def merge_dicts(dict_a: Dict[str, Any], dict_b: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries, combining their values."""
        merged = dict_a.copy()
        for key, value in dict_b.items():
            if key not in merged:
                merged[key] = value
                continue

            existing = merged[key]
            if isinstance(value, list):
                # If existing is not a list, convert it to one
                if not isinstance(existing, list):
                    existing = [existing] if existing is not None else []
                merged[key] = merge_lists(existing, value)
            elif isinstance(value, dict):
                # If existing is not a dict, create one with a default key
                if not isinstance(existing, dict):
                    existing = {"value": existing} if existing is not None else {}
                merged[key] = merge_dicts(existing, value)
            elif isinstance(existing, list):
                # Keep existing list structure, add new value
                merged[key] = merge_lists(
                    existing, [value] if value is not None else []
                )
            elif isinstance(existing, dict):
                # Keep existing dict structure, add new value under default key
                value_dict = {"value": value} if value is not None else {}
                merged[key] = merge_dicts(existing, value_dict)
            else:
                # Both are simple values, convert to list
                merged[key] = merge_lists(
                    [existing] if existing is not None else [],
                    [value] if value is not None else [],
                )

        return merged

    # Combine everything using the merge functions
    combined = ModelDetails(
        scientific_fields=merge_lists(a.scientific_fields, b.scientific_fields),
        preprocessing=merge_dicts(a.preprocessing, b.preprocessing),
        compute_resources=merge_dicts(a.compute_resources, b.compute_resources),
        training_platform=merge_dicts(a.training_platform, b.training_platform),
        architecture=merge_dicts(a.architecture, b.architecture),
        training_stack=merge_dicts(a.training_stack, b.training_stack),
        training_details=merge_dicts(a.training_details, b.training_details),
        dataset=merge_dicts(a.dataset, b.dataset),
        evaluation=merge_dicts(a.evaluation, b.evaluation),
        use_cases=merge_lists(a.use_cases, b.use_cases),
    )

    return combined


class ModelSummaryPipeline:
    """Pipeline for extracting structured information about foundation models from papers."""

    PIPELINE_NAME = "model_summary"

    def __init__(self, storage: LocalStorage):
        self.storage = storage
        self.config = get_config()
        self.manifest = ManifestManager()

    def get_project_results_dir(self, project_name: str) -> Path:
        """Get the directory for storing project-specific results."""
        return (
            self.storage.get_project_dir(project_name) / "results" / self.PIPELINE_NAME
        )

    def _get_project_results_path(self, project_name: str) -> Path:
        """Get the path for storing project results."""
        return (
            self.get_project_results_dir(project_name) / f"{self.PIPELINE_NAME}.jsonl"
        )

    def has_been_processed(self, project_id: str) -> bool:
        """Check if a project has already been processed.

        Args:
            project_id: ID of the project to check

        Returns:
            True if the project has been processed, False otherwise.
        """
        results_path = (
            self.get_project_results_dir(project_id) / f"{self.PIPELINE_NAME}.jsonl"
        )
        return results_path.exists()

    def _save_project_results(
        self, project_name: str, results: Dict[str, ModelDetails]
    ) -> None:
        """Save project results to disk."""
        results_path = self._get_project_results_path(project_name)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to dict for JSON serialization
        results_dict = {
            paper_path: details.model_dump() for paper_path, details in results.items()
        }

        logger.info(
            f"Saving model summaries for project {project_name} to {results_path}"
        )
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    def load_results(self, project_name: str) -> Dict[str, ModelDetails]:
        """
        Load saved results for a project.

        Args:
            project_name: Name or ID of the project

        Returns:
            Dictionary mapping paper paths to their extracted model details
        """
        results_path = self._get_project_results_path(project_name)
        if not results_path.exists():
            return {}

        try:
            with open(results_path) as f:
                results_dict = json.load(f)

            # Convert loaded dict back to ModelDetails objects
            return {
                paper_path: ModelDetails(**details)
                for paper_path, details in results_dict.items()
            }
        except Exception as e:
            logger.error(f"Error loading results for project {project_name}: {e}")
            return {}

    def process_paper(self, project_name: str, paper_path: Path) -> ModelDetails:
        """Process a single paper to extract model details."""
        # Read the paper text
        with open(paper_path, "r") as f:
            text = f.read()

        # Get OpenAI parameters from pipeline/model_summary config
        openai_params = get_openai_params(
            self.config.get("pipeline", {}).get("model_summary", {}).get("openai", {})
        )

        # Get model details via LLM with chunking
        response = get_structured_output(
            system_prompt=MODEL_SUMMARY_SYSTEM_PROMPT,
            user_prompt=MODEL_SUMMARY_USER_PROMPT.format(
                text=text, project_name=project_name
            ),
            response_model=ModelDetails,
            reduce_op=combine_model_details,
            **openai_params,
        )

        return response

    def process_project(self, project_name: str) -> Dict[str, ModelDetails]:
        """Process all papers for a project and extract model details.

        Args:
            project_name: Name of the project to process

        Returns:
            Dictionary mapping paper paths to their extracted model details
        """
        try:
            # Get project metadata from manifest
            project = self.manifest.get_project(project_name)

            results = {}
            # Process each paper listed in the manifest
            for paper in project["paths"]["papers"]:
                # Add .txt suffix to the path
                paper_path = Path(paper["path"]).with_suffix(".txt")

                # Check if the text file exists
                if not paper_path.exists():
                    logger.warning(f"Text file not found for paper: {paper_path}")
                    continue

                try:
                    model_details = self.process_paper(project_name, paper_path)
                    results[paper_path.name] = model_details
                except Exception as e:
                    logger.error(f"Error processing paper {paper_path}: {e}")

            if not results:
                logger.warning(
                    f"No papers were successfully processed for project {project_name}"
                )
            else:
                # Save project results
                self._save_project_results(project_name, results)

            return results

        except Exception as e:
            logger.error(f"Error accessing project {project_name}: {e}")
            return {}
