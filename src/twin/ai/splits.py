"""
Evaluation Protocol and Splitting Engine for Line 3 AI Models.
Implements:
  1. Forward-chaining (3 temporal folds across weeks 0 to 102)
  2. Leave-Scenario-Out (Scenarios A, B, C, D with configurable safety margin)
Ensures zero future-label leakage and zero cross-split contamination.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import yaml

UNITS_PER_WEEK = 16.75  # 168 hours / week assumption


def get_week_from_sim_time(sim_time: float) -> int:
    """Convert relative simulation time unit into estimated calendar week index."""
    return int(np.floor(sim_time / UNITS_PER_WEEK))


def get_forward_chaining_splits(
    df: pd.DataFrame,
    time_col: str = "decision_time",
    folds_config: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate train and validation indices for 3 forward-chaining temporal folds.
    Prevents temporal leakage by strictly maintaining chronological order.
    """
    if folds_config is None:
        folds_config = [
            {"fold": 1, "train_weeks": [0, 49], "val_weeks": [50, 64]},
            {"fold": 2, "train_weeks": [0, 64], "val_weeks": [65, 79]},
            {"fold": 3, "train_weeks": [0, 79], "val_weeks": [80, 102]},
        ]

    weeks = (df[time_col] / UNITS_PER_WEEK).astype(int)
    splits = []

    for cfg in folds_config:
        f_idx = cfg["fold"]
        tr_w0, tr_w1 = cfg["train_weeks"]
        va_w0, va_w1 = cfg["val_weeks"]

        train_mask = (weeks >= tr_w0) & (weeks <= tr_w1)
        val_mask = (weeks >= va_w0) & (weeks <= va_w1)

        splits.append({
            "fold": f_idx,
            "train_weeks": cfg["train_weeks"],
            "val_weeks": cfg["val_weeks"],
            "train_indices": df.index[train_mask].to_numpy(),
            "val_indices": df.index[val_mask].to_numpy(),
            "train_size": int(train_mask.sum()),
            "val_size": int(val_mask.sum()),
        })

    return splits


def get_leave_scenario_out_splits(
    df: pd.DataFrame,
    time_col: str = "decision_time",
    scenarios_config: Optional[Dict[str, Any]] = None,
    margin_units: float = 3.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate train and test splits for Leave-Scenario-Out hold-out models.
    For scenario S:
      Hold-out block = [t_start - margin, t_end + margin]
      Train = all data OUTSIDE hold-out block
      Test  = scenario window [t_start, t_end]
    """
    if scenarios_config is None:
        scenarios_config = {
            "A": {"t_start": 362.0, "t_end": 386.0},
            "B": {"t_start": 492.0, "t_end": 502.0},
            "C": {"t_start": 730.0, "t_end": 745.0},
            "D": {"t_start": 850.0, "t_end": 890.0},
        }

    splits = {}
    times = df[time_col].to_numpy()

    for sc_name, sc_info in scenarios_config.items():
        t0 = sc_info["t_start"]
        t1 = sc_info["t_end"]
        block_t0 = t0 - margin_units
        block_t1 = t1 + margin_units

        # Test set is the precise scenario window
        test_mask = (times >= t0) & (times <= t1)
        # Train set is everything strictly outside the safety margin block
        train_mask = (times < block_t0) | (times > block_t1)

        splits[sc_name] = {
            "scenario": sc_name,
            "t_start": t0,
            "t_end": t1,
            "margin_units": margin_units,
            "train_indices": df.index[train_mask].to_numpy(),
            "test_indices": df.index[test_mask].to_numpy(),
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
        }

    return splits
