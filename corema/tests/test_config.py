import pytest
from unittest.mock import patch
import os
from pathlib import Path

from ..config import get_config, get_project_root


def test_missing_config_file(tmp_path: Path) -> None:
    """Test behavior when config.yaml is missing."""
    with patch.dict(
        os.environ, {"COLLECTOR_CONFIG_PATH": str(tmp_path / "nonexistent.yaml")}
    ):
        with pytest.raises(FileNotFoundError):
            get_config()


def test_project_root() -> None:
    """Test that project root is correctly identified."""
    root = get_project_root()
    assert root.name == "corema"
    assert (root / "config.yaml").exists()
