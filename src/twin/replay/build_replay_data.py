"""
Builds offline replay datasets for Line 3 Digital Twin from raw exploration data.
Extracts chronological station visits, computes transit times, attaches ground truth
QC outcomes, and joins the 39 informative sensor features from train_numeric.csv.
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List, Set, Any
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import yaml

from twin.paths import (
    TWIN_CONFIG_PATH,
    REPLAY_DATA_DIR,
    EXPLORATION_OUTPUT_DIR,
    RAW_TRAIN_NUMERIC,
)

# Core Line 3 stations in order
LINE3_CORE_STATIONS = [29, 30, 33, 34, 35, 36, 37]
MINUTES_PER_UNIT = 601.7910447761194  # 168 hours / 16.75 units * 60 min


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    cfg_file = config_path if config_path is not None else str(TWIN_CONFIG_PATH)
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_scenario_data(
    t_start: float,
    t_end: float,
    output_name: str,
    output_dir: str = str(REPLAY_DATA_DIR),
    features_csv: str = str(EXPLORATION_OUTPUT_DIR / "recommended_subset_features_filtered.csv"),
    times_parquet: str = str(EXPLORATION_OUTPUT_DIR / "part_station_times.parquet"),
    response_parquet: str = str(EXPLORATION_OUTPUT_DIR / "part_response.parquet"),
    numeric_csv: str = str(RAW_TRAIN_NUMERIC),
):
    """Extract, enrich, and sort chronological events for a specified time window."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{output_name}.parquet")
    print(f"\n=======================================================")
    print(f"Building replay dataset: {output_name} [t={t_start} to {t_end}]")
    print(f"Output destination: {out_file}")

    t0 = time.time()

    # 1. Load feature mapping (which features belong to which station)
    feat_df = pd.read_csv(features_csv)
    feature_cols = feat_df["col_name"].tolist()
    station_feat_map: Dict[int, List[str]] = {s: [] for s in LINE3_CORE_STATIONS}
    for _, r in feat_df.iterrows():
        s_num = int(r["station"].replace("S", ""))
        if s_num in station_feat_map:
            station_feat_map[s_num].append(r["col_name"])

    print(f"Loaded {len(feature_cols)} informative features across Line 3 stations.")

    # 2. Filter visits from part_station_times.parquet
    print("Filtering Line 3 station visits from part_station_times.parquet...")
    table = pq.read_table(
        times_parquet,
        filters=[
            ("entry_time", ">=", t_start),
            ("entry_time", "<=", t_end),
            ("line", "==", 3),
        ],
    )
    df_visits = table.to_pandas()
    # Keep only core Line 3 stations (exclude S31 and auxiliary stations)
    df_visits = df_visits[df_visits["station"].isin(LINE3_CORE_STATIONS)].copy()
    print(f"Found {len(df_visits):,} visits across {df_visits['Id'].nunique():,} unique parts.")

    # 3. Load Response labels
    print("Attaching ground truth QC responses from part_response.parquet...")
    target_ids = set(df_visits["Id"].unique())
    df_resp = pd.read_parquet(response_parquet)
    df_resp = df_resp[df_resp["Id"].isin(target_ids)]
    resp_map = dict(zip(df_resp["Id"], df_resp["Response"]))

    # 4. Extract numeric features for targeted parts from train_numeric.csv
    print(f"Reading {len(feature_cols)} feature columns from {numeric_csv} in chunks...")
    chunks = []
    read_cols = ["Id"] + feature_cols
    chunk_size = 50000
    rows_matched = 0

    for chunk in pd.read_csv(numeric_csv, usecols=read_cols, chunksize=chunk_size):
        m = chunk[chunk["Id"].isin(target_ids)]
        if len(m) > 0:
            chunks.append(m)
            rows_matched += len(m)
        if rows_matched >= len(target_ids):
            break

    if chunks:
        df_features = pd.concat(chunks, ignore_index=True).set_index("Id")
    else:
        df_features = pd.DataFrame(columns=read_cols).set_index("Id")
    print(f"Extracted numeric features for {len(df_features):,} parts.")

    # 5. Process events per part to compute transit_minutes and assemble records
    print("Computing inter-station transit minutes and assembling events...")
    # Sort visits per part by entry_time, then station hierarchy
    station_order = {29: 1, 30: 2, 33: 3, 34: 4, 35: 5, 36: 5, 37: 6}
    df_visits["stn_rank"] = df_visits["station"].map(station_order).fillna(99)
    df_visits.sort_values(by=["Id", "stn_rank", "entry_time"], inplace=True)

    events = []
    parts_grouped = df_visits.groupby("Id")

    for part_id, group in parts_grouped:
        prev_entry = None
        qc_outcome = resp_map.get(part_id, 0)

        # Check if part has features
        has_feat_row = part_id in df_features.index
        part_feat_series = df_features.loc[part_id] if has_feat_row else None

        for _, row in group.iterrows():
            stn = int(row["station"])
            stn_id = f"S{stn}"
            entry = float(row["entry_time"])

            # Transit calculation
            if prev_entry is None or stn == 29:
                transit_min = 0.0
            else:
                transit_min = max(0.0, (entry - prev_entry) * MINUTES_PER_UNIT)
            prev_entry = entry

            # Extract features specific to this station
            stn_feats = {}
            if part_feat_series is not None:
                for fcol in station_feat_map.get(stn, []):
                    val = part_feat_series[fcol]
                    if pd.notna(val):
                        stn_feats[fcol] = float(val)

            is_entry = (stn == 29)
            is_exit = (stn == 37)

            events.append({
                "part_id": int(part_id),
                "station_id": stn_id,
                "station_num": stn,
                "stn_rank": station_order.get(stn, 99),
                "sim_time": entry,
                "sim_time_hours": round(entry * 10.02985, 4),
                "transit_minutes": round(transit_min, 4),
                "features_json": json.dumps(stn_feats),
                "is_entry": is_entry,
                "is_exit": is_exit,
                "response": int(qc_outcome) if is_exit else -1, # Revealed only at S37 exit
                "ground_truth_label": int(qc_outcome),
            })

    df_events = pd.DataFrame(events)

    # Sort strictly chronologically: sim_time first, then station sequence rank
    df_events.sort_values(by=["sim_time", "stn_rank", "part_id"], inplace=True)
    df_events.reset_index(drop=True, inplace=True)

    # Save to parquet
    df_events.to_parquet(out_file, index=False)
    elapsed = time.time() - t0
    print(f"Generated {len(df_events):,} chronological events in {elapsed:.2f}s")
    print(f"Saved: {out_file} (Size: {os.path.getsize(out_file) / (1024*1024):.2f} MB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Build real-data replay datasets for Line 3 Digital Twin.")
    parser.add_argument("--scenario", choices=["A", "B", "C", "D", "ALL"], default="ALL",
                        help="Replay scenario preset or ALL")
    parser.add_argument("--t-start", type=float, help="Custom window start time in units")
    parser.add_argument("--t-end", type=float, help="Custom window end time in units")
    parser.add_argument("--name", type=str, default="custom", help="Custom dataset name")
    args = parser.parse_args()

    cfg = load_config()
    scenarios_cfg = cfg["scenarios"]

    if args.t_start is not None and args.t_end is not None:
        build_scenario_data(args.t_start, args.t_end, args.name)
        return

    to_build = ["A", "B", "C", "D"] if args.scenario == "ALL" else [args.scenario]
    for sc in to_build:
        sc_info = scenarios_cfg[sc]
        build_scenario_data(sc_info["t_start"], sc_info["t_end"], f"scenario_{sc}")


if __name__ == "__main__":
    main()
