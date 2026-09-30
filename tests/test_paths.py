"""Test suite for centralized path resolution in twin.paths."""

import os
from pathlib import Path
import pytest
import yaml

from twin import paths


def test_project_root_detection():
    root = paths.find_project_root()
    assert root.exists()
    assert (root / "configs" / "paths.yaml").exists()
    assert (root / "pyproject.toml").exists()


def test_paths_yaml_validity():
    paths_file = paths.PROJECT_ROOT / "configs" / "paths.yaml"
    assert paths_file.exists()
    with open(paths_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "paths" in data
    p = data["paths"]
    assert "configs_dir" in p
    assert "twin_config" in p
    assert "artifacts_dir" in p
    assert "models_dir" in p
    assert "replay_data_dir" in p


def test_path_constants_and_helpers():
    assert paths.CONFIGS_DIR.exists()
    assert paths.TWIN_CONFIG_PATH.exists()
    assert paths.SCENARIOS_CONFIG_PATH.exists()
    assert paths.ARTIFACTS_DIR.exists()

    # Test get_path helper
    models_dir = paths.get_path("models_dir")
    assert models_dir == paths.MODELS_DIR

    with pytest.raises(KeyError):
        paths.get_path("non_existent_key_12345")


def test_ensure_dirs():
    paths.ensure_dirs()
    assert paths.ARTIFACTS_DIR.is_dir()
    assert paths.MODELS_DIR.is_dir()
    assert paths.REPLAY_DATA_DIR.is_dir()
    assert paths.AI_DATA_DIR.is_dir()
    assert paths.AI_OUTPUT_DIR.is_dir()
    assert paths.LOGS_DIR.is_dir()
