"""Information about OpenAI models and their limits.

References:
    - Model context window and output token limits: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    - GPT-3.5 Turbo specs: https://platform.openai.com/docs/models/gpt-3-5
"""

from dataclasses import dataclass


# Model name constants
GPT4_TURBO = "gpt-4-turbo-2024-04-09"
GPT4O = "gpt-4o-2024-08-06"
GPT4O_MINI = "gpt-4o-mini-2024-07-18"
GPT4_0613 = "gpt-4-0613"

# Default model to use
DEFAULT_MODEL = GPT4O_MINI


@dataclass
class ModelInfo:
    """Information about an OpenAI model."""

    context_window: int  # Maximum input tokens the model can process
    max_output_tokens: int  # Maximum tokens the model can generate in one response


OPENAI_MODELS = {
    # GPT-4o
    GPT4O: ModelInfo(
        context_window=128000,
        max_output_tokens=16384,
    ),
    # GPT-4o Mini
    GPT4O_MINI: ModelInfo(
        context_window=128000,
        max_output_tokens=16384,
    ),
    # GPT-4 Turbo
    GPT4_TURBO: ModelInfo(
        context_window=128000,
        max_output_tokens=4096,
    ),
    # GPT-4 0613
    GPT4_0613: ModelInfo(
        context_window=8192,
        max_output_tokens=8192,
    ),
}


def get_model_info(model: str) -> ModelInfo:
    """Get information about an OpenAI model.

    Args:
        model: The model name

    Returns:
        Information about the model

    Raises:
        ValueError: If the model is not supported
    """
    if model not in OPENAI_MODELS:
        raise ValueError(
            f"Model {model} not supported. Supported models: {', '.join(OPENAI_MODELS.keys())}"
        )
    return OPENAI_MODELS[model]
