"""
Phase A: Training Tables Builder for Bosch Line 3 AI Models.
Builds:
  1. Part-level defect prediction table (causal decision time at S35/S36 before S37 outcome)
  2. Window-level anomaly detection table (1 sim-hour windows across Line 3 stations, gap tagging)
Supports full dataset and --quick subsampling for fast experimentation.
"""

import os
import sys
import argparse
import time
import json
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from twin.paths import (
    TWIN_CONFIG_PATH,
    RAW_TRAIN_NUMERIC,
    AI_DATA_DIR,
    EXPLORATION_OUTPUT_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataBuild")

MINUTES_PER_UNIT = 601.7910447761194  # 168 hours / 16.75 units * 60 min
HOURS_PER_UNIT = 10.029850746268656
UNITS_PER_SIM_HOUR = 1.0 / HOURS_PER_UNIT  # ~0.0997 time units
UNITS_PER_WEEK = 16.75
CORE_STATIONS = [29, 30, 33, 34, 35, 36, 37]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_file = str(config_path) if config_path is not None else str(TWIN_CONFIG_PATH)
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_part_level_table(
    times_parquet: str = str(EXPLORATION_OUTPUT_DIR / "part_station_times.parquet"),
    response_parquet: str = str(EXPLORATION_OUTPUT_DIR / "part_response.parquet"),
    numeric_csv: str = str(RAW_TRAIN_NUMERIC),
    features_csv: str = str(EXPLORATION_OUTPUT_DIR / "recommended_subset_features_filtered.csv"),
    output_path: str = str(AI_DATA_DIR / "part_table.parquet"),
    quick_sample: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Builds the part-level dataset for defect prediction.
    Decision time = entry_time at S35 or S36.
    Computes inter-station transits and strictly causal context (no future leakage).
    """
    logger.info("Building part-level table for defect-risk modeling...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load feature columns and mapping
    feat_df = pd.read_csv(features_csv)
    feature_cols = feat_df["col_name"].tolist()
    logger.info(f"Loaded {len(feature_cols)} informative features across Line 3 stations.")

    # 2. Extract station visit timestamps from part_station_times.parquet
    logger.info("Loading Line 3 station timestamps...")
    table = pq.read_table(
        times_parquet,
        filters=[("station", "in", CORE_STATIONS), ("line", "==", 3)],
        columns=["Id", "station", "entry_time"],
    )
    df_visits = table.to_pandas()
    logger.info(f"Loaded {len(df_visits):,} raw visits.")

    # Pivot to get entry times per station for each part
    # A part in core flow must visit 29, 30, 33, 34, 37 AND either 35 or 36
    df_pivot = df_visits.pivot(index="Id", columns="station", values="entry_time")

    # Filter to core flow parts
    has_29 = df_pivot[29].notna()
    has_30 = df_pivot[30].notna()
    has_33 = df_pivot[33].notna()
    has_34 = df_pivot[34].notna()
    has_37 = df_pivot[37].notna()
    has_35 = df_pivot[35].notna()
    has_36 = df_pivot[36].notna()
    is_branch_35 = has_35 & (~has_36)
    is_branch_36 = has_36 & (~has_35)
    is_core = has_29 & has_30 & has_33 & has_34 & has_37 & (is_branch_35 | is_branch_36)

    df_core = df_pivot[is_core].copy()
    logger.info(f"Found {len(df_core):,} parts adhering strictly to core Line 3 flow.")

    # Attach Response label
    df_resp = pd.read_parquet(response_parquet).set_index("Id")
    df_core["Response"] = df_resp.loc[df_core.index, "Response"].fillna(0).astype(np.int8)

    # Branch flag: 0 for S35, 1 for S36
    df_core["branch"] = np.where(df_core[36].notna(), 1, 0).astype(np.int8)

    # Decision time = entry at branch station (S35 or S36)
    df_core["decision_time"] = np.where(df_core["branch"] == 1, df_core[36], df_core[35]).astype(np.float32)
    df_core["exit_time"] = df_core[37].astype(np.float32)
    df_core["week"] = (df_core["decision_time"] / UNITS_PER_WEEK).astype(np.int16)

    # Inter-station transit times (minutes)
    df_core["transit_29_30"] = np.maximum(0.0, (df_core[30] - df_core[29]) * MINUTES_PER_UNIT).astype(np.float32)
    df_core["transit_30_33"] = np.maximum(0.0, (df_core[33] - df_core[30]) * MINUTES_PER_UNIT).astype(np.float32)
    df_core["transit_33_34"] = np.maximum(0.0, (df_core[34] - df_core[33]) * MINUTES_PER_UNIT).astype(np.float32)
    df_core["transit_34_branch"] = np.maximum(0.0, (df_core["decision_time"] - df_core[34]) * MINUTES_PER_UNIT).astype(np.float32)

    # 3. Compute Causal Context Features (Strictly without future leakage)
    logger.info("Computing causal context features (throughput & rolling historical defect rate)...")
    # All S29 entry times sorted for fast binary search
    s29_times = np.sort(df_visits.loc[df_visits["station"] == 29, "entry_time"].to_numpy())
    decision_times = df_core["decision_time"].to_numpy()

    # S29 throughput in previous 1 sim-hour [t_decision - 1.0, t_decision]
    idx_right = np.searchsorted(s29_times, decision_times, side="right")
    idx_left = np.searchsorted(s29_times, decision_times - UNITS_PER_SIM_HOUR, side="left")
    df_core["s29_throughput_1h"] = (idx_right - idx_left).astype(np.float32)

    # Recent defect rate: last 500 parts whose S37 exit_time <= this part's decision_time
    # Sort all core parts by S37 exit_time
    sorted_exit_order = np.argsort(df_core["exit_time"].to_numpy())
    sorted_exit_times = df_core["exit_time"].to_numpy()[sorted_exit_order]
    sorted_exit_responses = df_core["Response"].to_numpy()[sorted_exit_order]
    cum_defects = np.concatenate([[0], np.cumsum(sorted_exit_responses)])

    baseline_p = 0.005069
    # Find how many parts finished before decision_time
    resolved_counts = np.searchsorted(sorted_exit_times, decision_times, side="right")
    start_counts = np.maximum(0, resolved_counts - 500)
    effective_window = resolved_counts - start_counts
    window_defects = cum_defects[resolved_counts] - cum_defects[start_counts]

    recent_p = np.where(
        effective_window > 0,
        window_defects / np.maximum(1, effective_window),
        baseline_p
    ).astype(np.float32)
    df_core["recent_defect_rate"] = recent_p

    # Reset index to have Id as column
    df_core = df_core.reset_index()

    # Subsample if quick mode requested
    if quick_sample is not None and quick_sample < len(df_core):
        logger.info(f"Quick mode: subsampling {quick_sample:,} parts (preserving all positives + random negatives)...")
        pos_df = df_core[df_core["Response"] == 1]
        neg_df = df_core[df_core["Response"] == 0]
        n_neg = max(1000, quick_sample - len(pos_df))
        neg_sample = neg_df.sample(n=min(len(neg_df), n_neg), random_state=random_seed)
        df_core = pd.concat([pos_df, neg_sample]).sort_values("decision_time").reset_index(drop=True)
        logger.info(f"Subsampled to {len(df_core):,} parts ({df_core['Response'].sum():,} positives).")

    # 4. Extract numeric features from train_numeric.csv in chunks
    target_ids = set(df_core["Id"].unique())
    logger.info(f"Reading {len(feature_cols)} features from {numeric_csv} for {len(target_ids):,} parts...")
    read_cols = ["Id"] + feature_cols
    chunks = []
    chunk_size = 50000
    matched_rows = 0

    for chunk in pd.read_csv(numeric_csv, usecols=read_cols, chunksize=chunk_size):
        m = chunk[chunk["Id"].isin(target_ids)]
        if len(m) > 0:
            chunks.append(m)
            matched_rows += len(m)
        if matched_rows >= len(target_ids):
            break

    if chunks:
        df_feats = pd.concat(chunks, ignore_index=True)
    else:
        df_feats = pd.DataFrame(columns=read_cols)

    logger.info(f"Extracted feature rows for {len(df_feats):,} parts.")

    # Merge features with df_core
    df_merged = pd.merge(df_core, df_feats, on="Id", how="left")

    # Add missing indicators for informative features (e.g. S35 features on S36 branch are NaN)
    for col in feature_cols:
        df_merged[f"{col}_isna"] = df_merged[col].isna().astype(np.int8)

    # Clean final column set
    base_cols = [
        "Id", "decision_time", "exit_time", "week", "branch",
        "transit_29_30", "transit_30_33", "transit_33_34", "transit_34_branch",
        "s29_throughput_1h", "recent_defect_rate"
    ]
    isna_cols = [f"{col}_isna" for col in feature_cols]
    final_cols = base_cols + feature_cols + isna_cols + ["Response"]

    # Reorder and sort chronologically by decision_time
    df_merged = df_merged[final_cols].sort_values("decision_time").reset_index(drop=True)

    # Save to parquet
    df_merged.to_parquet(output_path, index=False)
    elapsed = time.time() - t0
    logger.info(
        f"Part table built successfully: {len(df_merged):,} rows, "
        f"{df_merged['Response'].sum():,} positives ({df_merged['Response'].mean()*100:.3f}%) "
        f"in {elapsed:.1f}s -> {output_path}"
    )
    return df_merged


def build_window_level_table(
    times_parquet: str = str(EXPLORATION_OUTPUT_DIR / "part_station_times.parquet"),
    features_csv: str = str(EXPLORATION_OUTPUT_DIR / "recommended_subset_features_filtered.csv"),
    numeric_csv: str = str(RAW_TRAIN_NUMERIC),
    part_table_parquet: str = str(AI_DATA_DIR / "part_table.parquet"),
    output_path: str = str(AI_DATA_DIR / "window_table.parquet"),
    window_sim_hours: float = 1.0,
    min_parts_per_window: int = 20,
) -> pd.DataFrame:
    """
    Builds the 1-sim-hour window dataset for the LSTM anomaly autoencoder.
    Per window, per station: [max |z|, mean |z|, p90 transit, zero_delta_share, throughput].
    Marks windows with < 20 parts as gaps.
    """
    logger.info("Building window-level table for anomaly detection...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Window width in relative time units
    window_dt = window_sim_hours * UNITS_PER_SIM_HOUR

    # Load baseline stats to compute Z-scores
    feat_df = pd.read_csv(features_csv)
    station_feats: Dict[str, List[str]] = {f"S{s}": [] for s in CORE_STATIONS}
    feat_means = dict(zip(feat_df["col_name"], feat_df["mean"]))
    feat_stds = dict(zip(feat_df["col_name"], feat_df["std"]))
    for _, r in feat_df.iterrows():
        station_feats[r["station"]].append(r["col_name"])

    # Load part-level table (contains decision_time, transits, and features)
    df_parts = pd.read_parquet(part_table_parquet)
    t_min = float(df_parts["decision_time"].min())
    t_max = float(df_parts["decision_time"].max())
    logger.info(f"Time range: {t_min:.2f} to {t_max:.2f} units (~{(t_max-t_min)*10.03:.1f} hours).")

    # Generate continuous window grid
    window_edges = np.arange(np.floor(t_min), np.ceil(t_max) + window_dt, window_dt)
    n_windows = len(window_edges) - 1
    logger.info(f"Constructing {n_windows:,} 1-sim-hour temporal windows...")

    # Assign each part to a window index
    part_times = df_parts["decision_time"].to_numpy()
    window_indices = np.digitize(part_times, window_edges) - 1

    # Pre-compute feature absolute Z-scores per part
    for col in feat_df["col_name"]:
        m = feat_means[col]
        s = max(0.001, feat_stds[col])
        df_parts[f"{col}_abs_z"] = np.abs((df_parts[col] - m) / s)

    # Pre-compute station feature sets
    stn_abs_z_cols = {}
    for stn_id, fcols in station_feats.items():
        stn_abs_z_cols[stn_id] = [f"{col}_abs_z" for col in fcols]

    records = []
    # Load true all-parts S29 entry timestamps for accurate factory throughput & gap identification
    s29_table = pq.read_table(times_parquet, filters=[("station", "==", 29), ("line", "==", 3)], columns=["entry_time"])
    all_s29_times = s29_table["entry_time"].to_numpy()
    full_counts, _ = np.histogram(all_s29_times, bins=window_edges)

    gap_count = 0

    # Group parts by window
    df_parts["window_idx"] = window_indices
    grouped = df_parts.groupby("window_idx")

    for w_i in range(n_windows):
        t0_w = float(window_edges[w_i])
        t1_w = float(window_edges[w_i + 1])
        week = int(t0_w / UNITS_PER_WEEK)

        w_df = grouped.get_group(w_i) if w_i in grouped.groups else None
        part_count = int(full_counts[w_i])
        is_gap = (part_count < min_parts_per_window)
        if is_gap:
            gap_count += 1


        rec = {
            "window_id": w_i,
            "t_start": round(t0_w, 4),
            "t_end": round(t1_w, 4),
            "week": week,
            "part_count": part_count,
            "is_gap": is_gap,
        }

        # Station metrics: [max_abs_z, mean_abs_z, p90_transit, zero_delta_share, throughput]
        for s_num in CORE_STATIONS:
            stn_id = f"S{s_num}"
            z_cols = stn_abs_z_cols.get(stn_id, [])

            if not is_gap and w_df is not None:
                # Throughput
                throughput = float(part_count)

                # Sensor feature stats
                if z_cols:
                    # Filter parts visiting this station branch if applicable
                    if s_num == 35:
                        s_df = w_df[w_df["branch"] == 0]
                    elif s_num == 36:
                        s_df = w_df[w_df["branch"] == 1]
                    else:
                        s_df = w_df

                    if len(s_df) > 0 and len(z_cols) > 0:
                        vals = s_df[z_cols].values
                        valid_vals = vals[~np.isnan(vals)]
                        max_z = float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0
                        mean_z = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
                    else:
                        max_z, mean_z = 0.0, 0.0
                else:
                    max_z, mean_z = 0.0, 0.0

                # Transit pacing metrics
                if s_num == 30:
                    transit_vals = w_df["transit_29_30"].to_numpy()
                elif s_num == 33:
                    transit_vals = w_df["transit_30_33"].to_numpy()
                elif s_num == 34:
                    transit_vals = w_df["transit_33_34"].to_numpy()
                elif s_num in (35, 36):
                    branch_val = 0 if s_num == 35 else 1
                    transit_vals = w_df.loc[w_df["branch"] == branch_val, "transit_34_branch"].to_numpy()
                else:
                    transit_vals = np.array([0.0])

                if len(transit_vals) > 0:
                    p90_t = float(np.percentile(transit_vals, 90))
                    zero_share = float(np.mean(transit_vals <= 0.001))
                else:
                    p90_t, zero_share = 0.0, 1.0

            else:
                # In gap windows, fill with neutral zeros
                max_z, mean_z, p90_t, zero_share, throughput = 0.0, 0.0, 0.0, 1.0, 0.0

            rec[f"{stn_id}_max_abs_z"] = round(max_z, 4)
            rec[f"{stn_id}_mean_abs_z"] = round(mean_z, 4)
            rec[f"{stn_id}_p90_transit"] = round(p90_t, 4)
            rec[f"{stn_id}_zero_delta_share"] = round(zero_share, 4)
            rec[f"{stn_id}_throughput"] = round(throughput, 2)

        records.append(rec)

    df_windows = pd.DataFrame(records)
    df_windows.to_parquet(output_path, index=False)
    elapsed = time.time() - t0
    logger.info(
        f"Window table built: {len(df_windows):,} windows, "
        f"{gap_count:,} gaps ({gap_count/len(df_windows)*100:.1f}%), "
        f"{len(df_windows)-gap_count:,} valid training windows in {elapsed:.1f}s -> {output_path}"
    )
    return df_windows


def main():
    parser = argparse.ArgumentParser(description="Build training tables for Line 3 AI Models.")
    parser.add_argument("--quick", action="store_true", help="Quick mode (subsample parts for fast training)")
    parser.add_argument("--quick-samples", type=int, default=50000, help="Number of parts in quick mode")
    parser.add_argument("--config", type=str, default=str(TWIN_CONFIG_PATH), help="Path to twin config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_size = args.quick_samples if args.quick else None

    # 1. Build Part Table
    df_parts = build_part_level_table(quick_sample=sample_size)

    # 2. Build Window Table
    df_windows = build_window_level_table()

    # 3. Save summary stats
    stats = {
        "part_rows": len(df_parts),
        "positives": int(df_parts["Response"].sum()),
        "defect_rate_pct": float(df_parts["Response"].mean() * 100.0),
        "window_rows": len(df_windows),
        "gap_windows": int(df_windows["is_gap"].sum()),
        "valid_windows": int((~df_windows["is_gap"]).sum()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick_mode": bool(args.quick),
    }
    stats_file = str(AI_DATA_DIR / "dataset_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved dataset stats to {stats_file}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
