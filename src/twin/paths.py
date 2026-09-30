"""Centralized path management for the Bosch Digital Twin project.

Reads configs/paths.yaml and provides typed, reliable Path constants and helpers
resolved relative to the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def find_project_root() -> Path:
    """Find repository root by looking for pyproject.toml, configs/paths.yaml, or .git."""
    # Check environment variable first
    if "TWIN_PROJECT_ROOT" in os.environ:
        return Path(os.environ["TWIN_PROJECT_ROOT"]).resolve()

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "paths.yaml").exists() or (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()


def _load_paths_config() -> Dict[str, Any]:
    config_file = PROJECT_ROOT / "configs" / "paths.yaml"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("paths", {}) if data else {}
    return {}


_CONFIG = _load_paths_config()


def get_path(key: str, default: str | None = None) -> Path:
    """Get an absolute Path for a given path key defined in configs/paths.yaml."""
    val = _CONFIG.get(key, default)
    if val is None:
        raise KeyError(f"Path key '{key}' not found in configs/paths.yaml")
    return (PROJECT_ROOT / val).resolve()


# Core filesystem paths
RAW_DATA_DIR: Path = get_path("raw_data_dir", "data")
RAW_TRAIN_NUMERIC: Path = get_path("raw_train_numeric", "data/train_numeric.csv")
RAW_TRAIN_DATE: Path = get_path("raw_train_date", "data/train_date.csv")

CONFIGS_DIR: Path = get_path("configs_dir", "configs")
TWIN_CONFIG_PATH: Path = get_path("twin_config", "configs/twin_config.yaml")
SCENARIOS_CONFIG_PATH: Path = get_path("scenarios_config", "configs/scenarios.yaml")

ARTIFACTS_DIR: Path = get_path("artifacts_dir", "artifacts")
MODELS_DIR: Path = get_path("models_dir", "artifacts/models")
MODEL_REGISTRY_PATH: Path = get_path("model_registry", "artifacts/models/registry.json")
REPLAY_DATA_DIR: Path = get_path("replay_data_dir", "artifacts/replay_data")
AI_DATA_DIR: Path = get_path("ai_data_dir", "artifacts/ai_data")
AI_OUTPUT_DIR: Path = get_path("ai_output_dir", "artifacts/ai_output")
LOGS_DIR: Path = get_path("logs_dir", "artifacts/logs")

EXPLORATION_OUTPUT_DIR: Path = get_path("exploration_output_dir", "exploration/output")
DOCS_DIR: Path = get_path("docs_dir", "docs")


def ensure_dirs() -> None:
    """Ensure essential runtime artifact directories exist."""
    for d in [ARTIFACTS_DIR, MODELS_DIR, REPLAY_DATA_DIR, AI_DATA_DIR, AI_OUTPUT_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
