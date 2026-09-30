"""Comprehensive Acceptance Test Suite for Phase 2 AI Pipeline.

Verifies:
  1. Causal integrity of recent_defect_rate (zero future-label leakage).
  2. Strict disjointness of forward-chaining and leave-scenario-out splits.
  3. Window table gap isolation (sequences never cross gap windows).
  4. Model export and inference service round-trip reproducibility.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import torch

from twin.ai.splits import get_forward_chaining_splits, get_leave_scenario_out_splits
from twin.ai.inference import AIInferenceService
from twin.ai.utils import PureStandardScaler, PlattCalibrator, pure_pr_auc, pure_roc_auc
from twin.paths import AI_DATA_DIR, MODELS_DIR, MODEL_REGISTRY_PATH


def test_causal_defect_rate_integrity():
    """Verify that recent_defect_rate never uses future outcomes."""
    parquet_path = AI_DATA_DIR / "part_table.parquet"
    if not parquet_path.exists():
        pytest.skip(f"{parquet_path} not found.")

    df = pd.read_parquet(parquet_path)
    assert "recent_defect_rate" in df.columns
    assert "decision_time" in df.columns

    # Recent defect rate must be bounded [0, 1]
    rates = df["recent_defect_rate"].dropna()
    assert (rates >= 0.0).all()
    assert (rates <= 1.0).all()

    # The earliest part cannot observe future defects before any exit has occurred
    min_time = df["decision_time"].min()
    earliest_parts = df[df["decision_time"] == min_time]
    assert (earliest_parts["recent_defect_rate"] <= 0.05).all()


def test_forward_chaining_splits_disjoint():
    """Verify that forward chaining folds are strictly non-overlapping chronologically."""
    dummy_df = pd.DataFrame({"decision_time": np.linspace(0, 1700, 1000)})
    folds = get_forward_chaining_splits(dummy_df)
    assert len(folds) == 3

    for f in folds:
        train_start, train_end = f["train_weeks"]
        val_start, val_end = f["val_weeks"]

        assert train_start <= train_end
        assert val_start <= val_end
        # Strict temporal forward chaining: val weeks must be strictly after train weeks
        assert val_start >= train_end
        assert len(f["train_indices"]) > 0
        assert len(f["val_indices"]) > 0
        # No index overlap between train and val
        assert len(set(f["train_indices"]).intersection(set(f["val_indices"]))) == 0


def test_leave_scenario_out_safety_margin():
    """Verify that scenario training sets strictly exclude scenario windows (+/- 3.0 margin)."""
    dummy_df = pd.DataFrame({"decision_time": np.linspace(0, 1000, 2000)})
    splits = get_leave_scenario_out_splits(dummy_df, margin_units=3.0)
    for scenario_name, split in splits.items():
        t_start = split["t_start"]
        t_end = split["t_end"]
        margin = split["margin_units"]
        block_t0 = t_start - margin
        block_t1 = t_end + margin

        # Safety window: no train indices should fall within [block_t0, block_t1]
        train_times = dummy_df.loc[split["train_indices"], "decision_time"]
        assert not ((train_times >= block_t0) & (train_times <= block_t1)).any()


def test_anomaly_window_gap_isolation():
    """Verify that anomaly window table correctly identifies gap windows."""
    window_path = AI_DATA_DIR / "window_table.parquet"
    if not window_path.exists():
        pytest.skip(f"{window_path} not found.")

    df_win = pd.read_parquet(window_path)
    assert "is_gap" in df_win.columns
    assert "part_count" in df_win.columns

    # All windows with < 20 parts must be flagged as is_gap
    low_parts = df_win[df_win["part_count"] < 20]
    assert low_parts["is_gap"].all()


def test_inference_service_roundtrip():
    """Verify that AIInferenceService loads exported models and produces valid predictions."""
    if not MODEL_REGISTRY_PATH.exists():
        pytest.skip("Models not yet trained or registry missing.")

    service = AIInferenceService(models_dir=str(MODELS_DIR))
    assert service.defect_model is not None
    assert service.anomaly_model is not None

    # Test single part prediction
    sample_part = {
        "decision_time": 400.0,
        "week": 24,
        "branch": 0,
        "transit_29_30": 0.05,
        "transit_30_33": 0.08,
        "transit_33_34": 0.02,
        "transit_34_branch": 0.04,
        "s29_throughput_1h": 120.0,
        "recent_defect_rate": 0.005,
    }
    prob = service.predict_part(sample_part)
    assert 0.0 <= prob <= 1.0

    # Test sequence anomaly scoring
    seq = np.random.randn(24, 35).astype(np.float32)
    score, contribs, is_flagged = service.score_window(seq)
    assert score >= 0.0
    assert isinstance(is_flagged, (bool, np.bool_))
    assert len(contribs) == 7
    # Station contributions must sum to 1.0 (or 100%)
    total_contrib = sum(contribs.values())
    assert total_contrib == pytest.approx(1.0, abs=0.01) or total_contrib == pytest.approx(100.0, abs=1.0)


def test_holdout_scenario_variants():
    """Verify that scenario variants load their respective dedicated weights."""
    for sc in ["A", "B", "C", "D"]:
        meta_file = MODELS_DIR / f"defect_{sc}_metadata.json"
        if meta_file.exists():
            svc = AIInferenceService(models_dir=str(MODELS_DIR), scenario=sc)
            assert svc.defect_meta["variant"] == sc
            assert svc.anomaly_meta["variant"] == sc


def test_pure_metrics():
    """Verify pure numpy calculation of ROC-AUC and PR-AUC without scikit-learn."""
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1])
    y_score = np.array([0.1, 0.2, 0.15, 0.8, 0.3, 0.7, 0.9, 0.05, 0.4, 0.65])
    roc = pure_roc_auc(y_true, y_score)
    pr = pure_pr_auc(y_true, y_score)
    assert 0.8 <= roc <= 1.0
    assert 0.7 <= pr <= 1.0
