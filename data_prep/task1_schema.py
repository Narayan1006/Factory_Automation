"""
Task 1: Schema and Missing Value Analysis
-----------------------------------------
Dataset: Bosch Production Line Performance (train files ONLY)
1. Verify presence of 'Response' column in train files (confirms train vs test).
2. Count rows and columns per file (numeric, categorical, date).
3. Parse column naming convention: L<line>_S<station>_F<feature> or L<line>_S<station>_D<feature>.
4. Map which stations belong to which production line.
5. Compute exact missing-value rates per feature and aggregated per station.
6. Generate summary tables and visualizations in ./data_prep/output/.
"""

import os
import sys
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "data_prep/output"
FIGURES_DIR = "data_prep/output/figures"
CHUNKSIZE = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FILES = {
    "numeric": "train_numeric.csv",
    "categorical": "train_categorical.csv",
    "date": "train_date.csv",
}

def parse_col(col):
    """Parses column name into line, station, feature type, and feature index."""
    if col in ("Id", "Response"):
        return {"col_name": col, "line": None, "station": None, "feature_type": col, "feature_id": None}
    m = re.match(r"^L(\d+)_S(\d+)_([FD])(\d+)$", col)
    if m:
        line, station, ftype, fid = m.groups()
        return {
            "col_name": col,
            "line": int(line),
            "station": int(station),
            "feature_type": "Numeric/Cat" if ftype == "F" else "Date",
            "feature_id": int(fid),
        }
    return {"col_name": col, "line": None, "station": None, "feature_type": "Other", "feature_id": None}


def scan_numeric_or_date(file_label, file_name):
    """Reads numeric or date CSV in chunks of float32 using pandas."""
    fpath = os.path.join(DATA_DIR, file_name)
    print(f"\n[{file_label.upper()}] Scanning {file_name} in chunks of {CHUNKSIZE:,} (dtype=float32)...", flush=True)
    t0 = time.time()
    
    with open(fpath, "r") as f:
        header = f.readline().strip().split(",")
    
    col_counts = pd.Series(0, index=header, dtype=np.int64)
    total_rows = 0
    chunk_idx = 0
    
    for chunk in pd.read_csv(fpath, chunksize=CHUNKSIZE, dtype=np.float32, low_memory=False):
        chunk_idx += 1
        total_rows += len(chunk)
        col_counts += chunk.notna().sum()
        elapsed = time.time() - t0
        print(f"  Chunk {chunk_idx:02d}: processed {total_rows:>10,} rows in {elapsed:>6.1f}s", flush=True)
        
    print(f"  Finished {file_label}: {total_rows:,} rows x {len(header)} cols in {time.time()-t0:.1f}s", flush=True)
    
    df_stats = pd.DataFrame({
        "col_name": header,
        "file": file_label,
        "total_rows": total_rows,
        "missing_count": total_rows - col_counts.values,
        "non_null_count": col_counts.values,
        "missing_rate": (total_rows - col_counts.values) / total_rows,
    })
    return total_rows, df_stats


def scan_categorical(file_name):
    """
    Fast streaming line-by-line scanner for categorical CSV.
    Avoids Python string object allocation overhead by splitting raw string lines.
    """
    fpath = os.path.join(DATA_DIR, file_name)
    print(f"\n[CATEGORICAL] Scanning {file_name} line-by-line...", flush=True)
    t0 = time.time()
    
    with open(fpath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        num_cols = len(header)
        non_null_counts = np.zeros(num_cols, dtype=np.int64)
        total_rows = 0
        
        for line in f:
            total_rows += 1
            parts = line.strip().split(",")
            for i in range(num_cols):
                if parts[i] != "":
                    non_null_counts[i] += 1
            
            if total_rows % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {total_rows:>10,} rows in {elapsed:>6.1f}s", flush=True)
                
    print(f"  Finished categorical: {total_rows:,} rows x {num_cols} cols in {time.time()-t0:.1f}s", flush=True)
    
    df_stats = pd.DataFrame({
        "col_name": header,
        "file": "categorical",
        "total_rows": total_rows,
        "missing_count": total_rows - non_null_counts,
        "non_null_count": non_null_counts,
        "missing_rate": (total_rows - non_null_counts) / total_rows,
    })
    return total_rows, df_stats


def main():
    print("=" * 80, flush=True)
    print("TASK 1: SCHEMA AND MISSING VALUE ANALYSIS", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Verification of training files and labels
    print("\n[Step 1] Verifying data files and checking for 'Response' column...", flush=True)
    for flabel, fname in FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required data file: {fpath}")
        with open(fpath, "r") as f:
            header = f.readline().strip().split(",")
        has_id = "Id" in header
        has_resp = "Response" in header
        print(f"  File: {fname:<25} | Cols: {len(header):>5} | Id: {has_id} | Response: {has_resp}", flush=True)
        if flabel == "numeric":
            assert has_resp, "train_numeric.csv MUST contain the 'Response' column!"
            print("  >> Verified: 'Response' label is present in train_numeric.csv.", flush=True)
        else:
            assert not has_resp, f"{fname} should not have 'Response' column."

    # 2. Scan missing values and row counts across all 3 files
    print("\n[Step 2] Scanning column counts and missing values across all files...", flush=True)
    stats_list = []
    file_rows = {}
    
    # Numeric
    nrows_num, stats_num = scan_numeric_or_date("numeric", FILES["numeric"])
    file_rows["numeric"] = nrows_num
    stats_list.append(stats_num)
    
    # Date
    nrows_date, stats_date = scan_numeric_or_date("date", FILES["date"])
    file_rows["date"] = nrows_date
    stats_list.append(stats_date)
    
    # Categorical
    nrows_cat, stats_cat = scan_categorical(FILES["categorical"])
    file_rows["categorical"] = nrows_cat
    stats_list.append(stats_cat)
        
    all_stats = pd.concat(stats_list, ignore_index=True)
    
    # 3. Parse column names
    print("\n[Step 3] Parsing column names into Line, Station, and Feature...", flush=True)
    parsed_meta = pd.DataFrame([parse_col(c) for c in all_stats["col_name"]])
    merged = pd.concat([all_stats, parsed_meta.drop(columns=["col_name"])], axis=1)
    
    # Save column-level metadata
    merged.to_parquet(os.path.join(OUTPUT_DIR, "all_columns_metadata.parquet"))
    merged.to_csv(os.path.join(OUTPUT_DIR, "all_columns_metadata.csv"), index=False)
    print(f"  Saved metadata for all {len(merged):,} columns to {OUTPUT_DIR}/all_columns_metadata.csv", flush=True)

    # 4. Schema summary by file & feature type
    print("\n[Step 4] Schema breakdown by file and column type:", flush=True)
    summary_type = merged.groupby(["file", "feature_type"]).agg(
        col_count=("col_name", "count"),
        avg_missing_rate=("missing_rate", "mean")
    ).reset_index()
    print(summary_type.to_string(index=False), flush=True)

    # 5. Line to Station Mapping
    print("\n[Step 5] Mapping stations to production lines...", flush=True)
    station_feats = merged[merged["station"].notna()].copy()
    station_feats["station"] = station_feats["station"].astype(int)
    station_feats["line"] = station_feats["line"].astype(int)
    
    station_summary = station_feats.groupby(["line", "station"]).agg(
        num_numeric=("feature_type", lambda s: ((s == "Numeric/Cat") & (station_feats.loc[s.index, "file"] == "numeric")).sum()),
        num_categorical=("feature_type", lambda s: ((s == "Numeric/Cat") & (station_feats.loc[s.index, "file"] == "categorical")).sum()),
        num_date=("feature_type", lambda s: (s == "Date").sum()),
        total_features=("col_name", "count"),
        total_missing_cells=("missing_count", "sum"),
        total_non_null_cells=("non_null_count", "sum"),
        max_parts_at_station=("non_null_count", "max"),
        mean_parts_per_feature=("non_null_count", "mean"),
    ).reset_index()
    
    total_parts = file_rows["numeric"]
    station_summary["total_possible_cells"] = station_summary["total_features"] * total_parts
    station_summary["cell_missing_rate"] = station_summary["total_missing_cells"] / station_summary["total_possible_cells"]
    station_summary["station_part_coverage"] = station_summary["max_parts_at_station"] / total_parts
    
    station_summary.sort_values(by=["line", "station"], inplace=True)
    station_summary.to_csv(os.path.join(OUTPUT_DIR, "station_missingness_summary.csv"), index=False)
    print(f"  Saved station summary to {OUTPUT_DIR}/station_missingness_summary.csv", flush=True)
    
    # Print stations per line
    for line_id, grp in station_summary.groupby("line"):
        st_list = grp["station"].tolist()
        print(f"  Line {line_id}: {len(st_list)} stations -> S{min(st_list)} to S{max(st_list)}: {st_list}", flush=True)

    # Verify if any station belongs to multiple lines
    multi_line = station_feats.groupby("station")["line"].nunique()
    shared = multi_line[multi_line > 1]
    if len(shared) == 0:
        print("  >> Verification: Every station strictly belongs to exactly ONE line.", flush=True)
    else:
        print(f"  >> Notice: Stations belonging to multiple lines: {shared.to_dict()}", flush=True)

    # 6. Visualizations
    print("\n[Step 6] Generating figures...", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Features per station broken down by type
    x_labels = [f"S{s}\n(L{l})" for s, l in zip(station_summary["station"], station_summary["line"])]
    indices = np.arange(len(station_summary))
    width = 0.6
    
    axes[0].bar(indices, station_summary["num_numeric"], width, label="Numeric", color="#1f77b4")
    axes[0].bar(indices, station_summary["num_categorical"], width, bottom=station_summary["num_numeric"], label="Categorical", color="#ff7f0e")
    axes[0].bar(indices, station_summary["num_date"], width, 
                bottom=station_summary["num_numeric"] + station_summary["num_categorical"], 
                label="Date", color="#2ca02c")
    
    axes[0].set_title("Task 1: Feature Count per Station by Type (Numeric, Categorical, Date)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Station (Line)")
    axes[0].set_ylabel("Number of Features")
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(x_labels, rotation=90, fontsize=7)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Missing Rate per Station
    axes[1].bar(indices, station_summary["cell_missing_rate"] * 100, width, color="#d62728", alpha=0.85, edgecolor="black")
    axes[1].set_title("Task 1: Overall Missing Value Rate (%) per Station", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Station (Line)")
    axes[1].set_ylabel("Missing Value Percentage (%)")
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(x_labels, rotation=90, fontsize=7)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "task1_schema_and_missingness.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("TASK 1 FINDINGS SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Parts (Rows):           {total_parts:,}", flush=True)
    print(f"train_numeric.csv:            {file_rows['numeric']:,} rows, {len(all_stats[all_stats['file']=='numeric'])} cols (Id, 968 numeric, Response)", flush=True)
    print(f"train_categorical.csv:        {file_rows['categorical']:,} rows, {len(all_stats[all_stats['file']=='categorical'])} cols (Id, 2,140 categorical)", flush=True)
    print(f"train_date.csv:               {file_rows['date']:,} rows, {len(all_stats[all_stats['file']=='date'])} cols (Id, 1,156 date)", flush=True)
    print(f"Total Columns Across Files:   {len(all_stats):,} columns", flush=True)
    print(f"Total Distinct Stations:      {len(station_summary)} stations (S0 to S51)", flush=True)
    print(f"Lines Breakdown:", flush=True)
    for l_id, grp in station_summary.groupby("line"):
        print(f"  - Line {l_id}: {len(grp)} stations, {grp['total_features'].sum()} total features", flush=True)
    print(f"Average Cell Missingness:     {station_summary['cell_missing_rate'].mean()*100:.2f}% (Reflects routed assembly paths)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
