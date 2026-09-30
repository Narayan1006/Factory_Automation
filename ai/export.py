"""
Phase D: Model Export and Registry Manager.
Serializes trained defect MLPs, calibrators, scalers, and LSTM autoencoders.
Maintains models/registry.json for Leave-Scenario-Out hold-out models and general production models.
"""

import os
import sys
import json
import hashlib
import time
import subprocess
from typing import Dict, List, Any, Optional
import torch

from ai.defect_model import DefectMLP
from ai.anomaly_model import LSTMAutoencoder
from ai.utils import PureStandardScaler, PlattCalibrator


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for auditability and lineage tracking."""
    if not os.path.exists(filepath):
        return "not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_git_commit() -> str:
    """Returns current git commit hash if available, else 'unversioned'."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit[:10]
    except Exception:
        return "unversioned"


def export_defect_model(
    model: DefectMLP,
    scaler: PureStandardScaler,
    calibrator: PlattCalibrator,
    threshold: float,
    feature_names: List[str],
    train_medians: Dict[str, float],
    metrics: Dict[str, Any],
    variant_name: str,
    output_dir: str = "models",
    data_path: str = "ai/data/part_table.parquet",
) -> str:
    """Saves defect model artifact bundle."""
    os.makedirs(output_dir, exist_ok=True)
    base_prefix = os.path.join(output_dir, f"defect_{variant_name}")

    # 1. Weights
    weights_path = f"{base_prefix}_weights.pt"
    torch.save(model.state_dict(), weights_path)

    # 2. Scaler & Calibrator (Portable JSON dicts)
    scaler_path = f"{base_prefix}_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f)

    calibrator_path = f"{base_prefix}_calibrator.json"
    with open(calibrator_path, "w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f)

    # 3. Metadata & Metrics
    meta = {
        "variant": variant_name,
        "type": "DefectMLP",
        "in_features": len(feature_names),
        "feature_names": feature_names,
        "train_medians": train_medians,
        "optimal_threshold": float(threshold),
        "metrics": metrics,
        "weights_file": os.path.basename(weights_path),
        "scaler_file": os.path.basename(scaler_path),
        "calibrator_file": os.path.basename(calibrator_path),
        "data_hash": compute_file_hash(data_path),
        "git_commit": get_git_commit(),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = f"{base_prefix}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta_path


def export_anomaly_model(
    model: LSTMAutoencoder,
    scaler: PureStandardScaler,
    threshold: float,
    feature_names: List[str],
    metrics: Dict[str, Any],
    variant_name: str,
    output_dir: str = "models",
    data_path: str = "ai/data/window_table.parquet",
) -> str:
    """Saves anomaly autoencoder artifact bundle."""
    os.makedirs(output_dir, exist_ok=True)
    base_prefix = os.path.join(output_dir, f"anomaly_{variant_name}")

    # 1. Weights
    weights_path = f"{base_prefix}_weights.pt"
    torch.save(model.state_dict(), weights_path)

    # 2. Scaler (JSON dict)
    scaler_path = f"{base_prefix}_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f)

    # 3. Metadata & Metrics
    meta = {
        "variant": variant_name,
        "type": "LSTMAutoencoder",
        "in_features": len(feature_names),
        "feature_names": feature_names,
        "threshold": float(threshold),
        "metrics": metrics,
        "weights_file": os.path.basename(weights_path),
        "scaler_file": os.path.basename(scaler_path),
        "data_hash": compute_file_hash(data_path),
        "git_commit": get_git_commit(),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = f"{base_prefix}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta_path


def update_registry(
    registry_path: str = "models/registry.json",
    scenarios: List[str] = ["A", "B", "C", "D"],
):
    """Generates models/registry.json mapping scenario -> holdout model version."""
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    reg = {
        "active_general": {
            "defect_model": "defect_general",
            "anomaly_model": "anomaly_general",
            "description": "General model for production outside held-out scenarios",
        },
        "scenarios": {},
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for sc in scenarios:
        reg["scenarios"][sc] = {
            "defect_model": f"defect_{sc}",
            "anomaly_model": f"anomaly_{sc}",
            "description": f"Strict hold-out variant: trained on data excluding Scenario {sc}",
        }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

    return reg
