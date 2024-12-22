import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import dotenv_values, find_dotenv
import logging
from .utils.openai_models import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root directory
    """
    return Path(__file__).parent.parent


def get_config() -> Dict[str, Any]:
    """Get configuration by merging config.yaml and environment variables.
    Environment variables from .env take precedence over config.yaml values.

    Returns:
        Dictionary containing merged configuration
    """
    # Load config file
    config_path = Path(
        os.getenv("COLLECTOR_CONFIG_PATH", str(get_project_root() / "config.yaml"))
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load environment variables from .env file
    dotenv_path = find_dotenv()
    if not dotenv_path:
        logger.warning("No .env file found")
    else:
        env_vars = dotenv_values(dotenv_path)
        # Environment variables take precedence over config file
        config.update(env_vars)

    return config  # type: ignore[no-any-return]


def get_openai_params(config: dict) -> dict:
    """Get OpenAI parameters from config with defaults.

    Args:
        config: Config dictionary containing OpenAI parameters

    Returns:
        Dictionary of OpenAI parameters with the following keys:
        - chunk_size: Maximum tokens per chunk (default: 8192)
        - model: Model to use (default: DEFAULT_MODEL)
        - temperature: Sampling temperature (default: 0.0)
        - max_tokens: Maximum tokens to generate (default: None)
        - overlap_tokens: Number of tokens to overlap between chunks (default: 100)
    """
    return {
        "chunk_size": config.get(
            "chunk_size", 8192
        ),  # Default to 8K tokens if not specified
        "model": config.get("model", DEFAULT_MODEL),
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens"),
        "overlap_tokens": config.get("overlap_tokens", 100),
    }
