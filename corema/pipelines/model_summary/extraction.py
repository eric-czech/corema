"""Module for extracting model summary information from scientific papers."""

from typing import List, Dict, Any
from pathlib import Path
import json
from pydantic import BaseModel, Field
import logging

from corema.storage import LocalStorage
from corema.utils.openai_api import get_structured_output
from corema.pipelines.common import PipelineBase, get_openai_pipeline_params

logger = logging.getLogger(__name__)


MODEL_SUMMARY_SYSTEM_PROMPT = """You are an expert at analyzing scientific papers about foundation models and extracting structured information about their training, architecture, and evaluation details.
Your task is to extract detailed technical information ONLY about the primary model being introduced in the paper.
Do not include information about baseline models, comparison models, or models used for context.
Extract only factual information that is explicitly stated in the text about the main model being presented.
If information for a field is not available, use empty arrays for lists.
Be precise and comprehensive in extracting technical details.

Important constraints:
- Extract information ONLY about the primary model being introduced
- No array should contain more than 10 items. If there are more, select only the most relevant/representative items.
- No string value should be longer than 100 characters. If longer, truncate while preserving key information.

Return your response as a JSON object with the following structure:
- scientific_fields: array of strings (max 10 items)
- preprocessing: object with:
  - tools: array of strings (max 10 items)
  - systems: array of strings (max 10 items)
  - data_cleaning: array of strings (max 10 items)
  - data_transformation: array of strings (max 10 items)
  - data_augmentation: array of strings (max 10 items)
  - normalization: array of strings (max 10 items)
  - tokenization: array of strings (max 10 items)
  - other_tools: array of strings (max 10 items)
- compute_resources: object with:
  - training_time: array of strings (max 10 items)
  - cost_estimate: array of strings (max 10 items)
  - gpu_count: array of integers (max 10 items)
  - gpu_type: array of strings (max 10 items)
  - other_hardware: array of strings (max 10 items)
- training_platform: object with:
  - cloud_provider: array of strings (max 10 items)
  - hpc_system: array of strings (max 10 items)
  - training_service: array of strings (max 10 items)
  - other_platforms: array of strings (max 10 items)
- architecture: object with:
  - model_type: array of strings (max 10 items)
  - parameter_count: array of integers (max 10 items)
  - architecture_type: array of strings (max 10 items)
  - key_components: array of strings (max 10 items)
- training_stack: object with:
  - frameworks: array of strings (max 10 items)
  - libraries: array of strings (max 10 items)
  - languages: array of strings (max 10 items)
  - tools: array of strings (max 10 items)
- training_details: object with:
  - parallelization: array of strings (max 10 items)
  - checkpointing: array of strings (max 10 items)
  - optimization_methods: array of strings (max 10 items)
  - regularization: array of strings (max 10 items)
  - loss_functions: array of strings (max 10 items)
  - training_techniques: array of strings (max 10 items)
- dataset: object with:
  - size: array of strings (max 10 items)
  - modalities: array of strings (max 10 items)
- evaluation: object with:
  - datasets: array of strings (max 10 items)
  - metrics: array of strings (max 10 items)
  - key_results: object with string keys and array values (max 10 items per array)
- use_cases: array of strings (max 10 items)"""

MODEL_SUMMARY_USER_PROMPT = """The paper describes the {project_name} model. Extract structured information ONLY about this primary foundation model being introduced in the paper.
Focus only on technical details related to the architecture, training process, computational resources, and evaluation of the {project_name} model.
Do not include information about baseline models, comparison models, or models referenced for context.

If information is not clearly stated, use null for scalar values or empty arrays for lists.

Paper text:
```
{text}
```

Return your response as a JSON object."""


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


class ModelSummaryPipeline(PipelineBase[ModelDetails]):
    """Pipeline for extracting structured information about foundation models from papers."""

    PIPELINE_NAME = "model_summary"

    def __init__(self, storage: LocalStorage):
        super().__init__(self.PIPELINE_NAME, storage)

    def _flatten_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten model details into a list of records."""
        flattened = []
        for paper_path, details in results.items():
            record = {"paper_path": paper_path, **details}
            flattened.append(record)
        return flattened

    def process_paper(
        self, project_id: str, project_name: str, paper_path: Path
    ) -> ModelDetails:
        """Process a single paper to extract model details.

        Args:
            project_id: ID of the project
            project_name: Name of the project (used in prompts)
            paper_path: Path to the paper text file
        """
        # Read the paper text
        with open(paper_path, "r") as f:
            text = f.read()

        # Get OpenAI parameters from pipeline config
        openai_params = get_openai_pipeline_params("model_summary")

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

    def load_results(self, project_id: str) -> Dict[str, ModelDetails]:
        """Load project results from disk.

        Args:
            project_id: ID of the project to load results for

        Returns:
            Dictionary mapping paper paths to their extracted model details
        """
        results_path = self.get_project_results_path(project_id)
        if not results_path.exists():
            return {}

        with open(results_path) as f:
            results_dict = json.load(f)
            return {
                path: ModelDetails.model_validate(details)
                for path, details in results_dict.items()
            }
