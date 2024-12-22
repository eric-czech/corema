"""String templates for model summary pipeline prompts."""

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
