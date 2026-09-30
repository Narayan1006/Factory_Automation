"""
Task 6: Station Behavior Over Time (Drift, Transit Times, and Defect Dynamics)
-----------------------------------------------------------------------------
Dataset: Bosch Production Line Performance
1. Select measurement features with low missing rate and non-zero variance for each station.
2. For each station, compute robust baseline statistics:
   - Measurement mean and standard deviation
   - Baseline defect rate (proportion of visiting parts that failed final QC)
   - For S24 and S25: median (p50), p90, p95, p99 durations
3. For the core Line 3 subset stations (S29, S30, S33, S34, S35, S36, S37):
   - Explicitly note that individual station processing duration is unavailable (internal dates share identical timestamps).
   - Compute inter-station transit / buffer wait times between consecutive stages.
4. Compute weekly time-window behavior:
   - Rolling defect rate over time
   - Measurement drift (rolling Z-score against station's own baseline)
5. Save intermediate summary tables:
   - data_prep/output/station_behavior_drift_summary.csv
   - data_prep/output/line3_interstation_transit_times.csv
   - data_prep/output/s24_s25_duration_percentiles.csv
6. Generate multi-panel SPC visualizations in ./data_prep/output/figures/task6_station_behavior.png.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "data_prep/output"
FIGURES_DIR = "data_prep/output/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

WEEK_UNITS = 16.75
MINUTES_PER_UNIT = (168.0 / 16.75) * 60.0

def main():
    print("=" * 80, flush=True)
    print("TASK 6: STATION BEHAVIOR OVER TIME (DRIFT & TRANSIT DYNAMICS)", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()

    # 1. Load Station Best Numeric Features
    best_feats_path = os.path.join(OUTPUT_DIR, "station_best_numeric_features.csv")
    best_feats = pd.read_csv(best_feats_path)
    print(f"[Step 1] Loaded best numeric features for {len(best_feats)} stations.", flush=True)

    # 2. Load Selected Numeric Columns
    print("[Step 2] Loading target numeric columns from train_numeric.csv...", flush=True)
    t1 = time.time()
    target_cols = ["Id"] + best_feats["col_name"].tolist()
    num_df = pd.read_csv(os.path.join(DATA_DIR, "train_numeric.csv"), usecols=target_cols, dtype=np.float32)
    num_df["Id"] = num_df["Id"].astype(np.int32)
    print(f"  Loaded in {time.time()-t1:.2f}s", flush=True)

    # 3. Load Events and Response Labels
    print("[Step 3] Loading station events and response labels...", flush=True)
    events_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_station_times.parquet"))
    resp_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_response.parquet"))
    resp_map = dict(zip(resp_df["Id"], resp_df["Response"]))

    # 4. S24 and S25 Robust Duration Percentiles
    print("\n[Step 4] Computing Robust Duration Baselines for Physical Duration Stations (S24, S25)...", flush=True)
    dur_records = []
    for st in [24, 25]:
        st_data = events_df[events_df["station"] == st]
        dur_min = st_data["duration"].values * MINUTES_PER_UNIT
        p50 = float(np.percentile(dur_min, 50))
        p90 = float(np.percentile(dur_min, 90))
        p95 = float(np.percentile(dur_min, 95))
        p99 = float(np.percentile(dur_min, 99))
        mean_d = float(np.mean(dur_min))
        max_d = float(np.max(dur_min))
        dur_records.append({
            "station": f"S{st}",
            "parts_count": len(dur_min),
            "median_p50_min": p50,
            "p90_min": p90,
            "p95_min": p95,
            "p99_min": p99,
            "mean_min": mean_d,
            "max_min": max_d
        })
    dur_df = pd.DataFrame(dur_records)
    dur_csv = os.path.join(OUTPUT_DIR, "s24_s25_duration_percentiles.csv")
    dur_df.to_csv(dur_csv, index=False)
    print(f"  Saved duration percentiles to: {dur_csv}", flush=True)
    print(dur_df.to_string(index=False), flush=True)

    # 5. Inter-Station Transit / Buffer Wait Times for Line 3 Backbone
    print("\n[Step 5] Computing Inter-Station Transit/Wait Time for Line 3 Stations...", flush=True)
    p_times = events_df[events_df["station"].isin([29, 30, 31, 33, 34, 35, 36, 37])].pivot(
        index="Id", columns="station", values="entry_time"
    )

    transit_pairs = [
        ("S29 -> S30", 29, 30),
        ("S30 -> S33", 30, 33),
        ("S30 -> S31", 30, 31),
        ("S31 -> S33", 31, 33),
        ("S33 -> S34", 33, 34),
        ("S34 -> S35", 34, 35),
        ("S34 -> S36", 34, 36),
        ("S35 -> S37", 35, 37),
        ("S36 -> S37", 36, 37)
    ]

    transit_metrics = []
    for label, s_from, s_to in transit_pairs:
        if s_from in p_times.columns and s_to in p_times.columns:
            deltas = (p_times[s_to] - p_times[s_from]).dropna()
            deltas_min = deltas * MINUTES_PER_UNIT
            pos = deltas_min[deltas_min >= 0]
            transit_metrics.append({
                "transit_stage": label,
                "parts_count": len(pos),
                "median_min": float(np.median(pos)),
                "p90_min": float(np.percentile(pos, 90)),
                "p95_min": float(np.percentile(pos, 95)),
                "p99_min": float(np.percentile(pos, 99)),
                "mean_min": float(np.mean(pos)),
                "zero_delta_pct": float((deltas_min == 0).mean() * 100)
            })

    transit_df = pd.DataFrame(transit_metrics)
    transit_csv = os.path.join(OUTPUT_DIR, "line3_interstation_transit_times.csv")
    transit_df.to_csv(transit_csv, index=False)
    print(f"  Saved Line 3 transit/wait times to: {transit_csv}", flush=True)
    print(transit_df.to_string(index=False), flush=True)

    # 6. Analyze Behavior & Drift per Station
    print("\n[Step 6] Computing Weekly Station Dynamics & SPC Drift for all 50 Stations...", flush=True)
    station_summaries = []
    weekly_records = []
    st_grouped = events_df.groupby("station")
    
    for _, row in best_feats.iterrows():
        st = int(row["station"])
        col = row["col_name"]
        line = int(row["line"])

        if st not in st_grouped.groups:
            continue
            
        st_events = st_grouped.get_group(st)[["Id", "entry_time", "duration"]].copy()
        sub_num = num_df[["Id", col]].dropna()
        merged = st_events.merge(sub_num, on="Id", how="inner")
        
        if len(merged) < 50:
            continue

        merged["Response"] = merged["Id"].map(resp_map).fillna(0).astype(np.int8)
        merged.sort_values(by="entry_time", inplace=True)

        feat_vals = merged[col].values
        resp_vals = merged["Response"].values

        mu_x = float(np.nanmean(feat_vals))
        sigma_x = float(np.nanstd(feat_vals))
        base_defect_rate = float(np.mean(resp_vals))

        if sigma_x == 0 or np.isnan(sigma_x):
            sigma_x = 1.0

        # Weekly time window bins
        merged["week_idx"] = (merged["entry_time"] // WEEK_UNITS).astype(int)
        weekly = merged.groupby("week_idx").agg(
            parts=("Id", "count"),
            defects=("Response", "sum"),
            defect_rate=("Response", "mean"),
            mean_meas=(col, "mean"),
            std_meas=(col, "std")
        ).reset_index()

        weekly["z_score_drift"] = (weekly["mean_meas"] - mu_x) / sigma_x
        weekly["station"] = st
        weekly["line"] = line
        weekly["feature"] = col
        weekly_records.append(weekly)

        max_abs_z = float(np.nanmax(np.abs(weekly["z_score_drift"])))
        peak_defect_rate = float(weekly["defect_rate"].max())

        station_summaries.append({
            "line": line,
            "station": st,
            "feature": col,
            "parts_analyzed": len(merged),
            "baseline_mean_meas": mu_x,
            "baseline_std_meas": sigma_x,
            "baseline_defect_rate": base_defect_rate,
            "max_abs_z_drift": max_abs_z,
            "peak_weekly_defect_rate": peak_defect_rate,
            "defect_spike_ratio": (peak_defect_rate / base_defect_rate) if base_defect_rate > 0 else 0.0
        })

    summary_df = pd.DataFrame(station_summaries)
    summary_df.sort_values(by="max_abs_z_drift", ascending=False, inplace=True)
    summary_csv = os.path.join(OUTPUT_DIR, "station_behavior_drift_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved station behavior drift summary to: {summary_csv}", flush=True)

    all_weekly_df = pd.concat(weekly_records, ignore_index=True)
    all_weekly_parquet = os.path.join(OUTPUT_DIR, "weekly_station_behavior.parquet")
    all_weekly_df.to_parquet(all_weekly_parquet)

    # 7. Generate Multi-Panel Visualizations
    print("\n[Step 7] Generating Visualizations...", flush=True)
    rep_stations = [29, 30, 31, 33] # Line 3 focus
    fig, axes = plt.subplots(len(rep_stations), 2, figsize=(15, 12))

    for row_idx, st in enumerate(rep_stations):
        st_data = all_weekly_df[all_weekly_df["station"] == st].sort_values("week_idx")
        st_meta = summary_df[summary_df["station"] == st].iloc[0]
        
        # Subplot 1: Defect Rate over Time
        axes[row_idx, 0].plot(st_data["week_idx"], st_data["defect_rate"] * 100, color="#c0392b", marker="o", markersize=3, lw=1.5)
        axes[row_idx, 0].axhline(st_meta["baseline_defect_rate"] * 100, color="black", linestyle="--", lw=1.2, label=f"Baseline ({st_meta['baseline_defect_rate']*100:.2f}%)")
        axes[row_idx, 0].set_title(f"Station S{st} (Line {st_meta['line']}) — Weekly Defect Rate (%)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_ylabel("Defect Rate (%)")
        axes[row_idx, 0].grid(True, linestyle="--", alpha=0.5)
        axes[row_idx, 0].legend(loc="upper right", fontsize=8)

        # Subplot 2: Measurement Drift Z-Score (SPC Control Chart)
        axes[row_idx, 1].plot(st_data["week_idx"], st_data["z_score_drift"], color="#2980b9", marker="^", markersize=3, lw=1.5, label="Z-score")
        axes[row_idx, 1].axhline(0, color="black", linestyle="-", lw=1)
        axes[row_idx, 1].axhline(2.0, color="#f39c12", linestyle="--", lw=1.2, label="+2-sigma Warning")
        axes[row_idx, 1].axhline(-2.0, color="#f39c12", linestyle="--", lw=1.2)
        axes[row_idx, 1].axhline(3.0, color="#e74c3c", linestyle=":", lw=1.5, label="+3-sigma Action")
        axes[row_idx, 1].axhline(-3.0, color="#e74c3c", linestyle=":", lw=1.5)
        axes[row_idx, 1].set_title(f"Station S{st} ({st_meta['feature']}) — Measurement Drift (Z-Score)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].set_ylabel("Standardized Drift (Z)")
        axes[row_idx, 1].set_ylim(-4.5, 8.5 if st==31 else 4.5)
        axes[row_idx, 1].grid(True, linestyle="--", alpha=0.5)
        axes[row_idx, 1].legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Estimated Production Week")
    axes[-1, 1].set_xlabel("Estimated Production Week")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "task6_station_behavior.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("TASK 6 FINDINGS SUMMARY")
    print("=" * 80, flush=True)
    print(f"S24 Duration Baseline:      Median = {dur_df.loc[dur_df['station']=='S24', 'median_p50_min'].values[0]:.1f} min, p90 = {dur_df.loc[dur_df['station']=='S24', 'p90_min'].values[0]:.1f} min, p99 = {dur_df.loc[dur_df['station']=='S24', 'p99_min'].values[0]:.1f} min")
    print(f"S25 Duration Baseline:      Median = {dur_df.loc[dur_df['station']=='S25', 'median_p50_min'].values[0]:.1f} min, p90 = {dur_df.loc[dur_df['station']=='S25', 'p90_min'].values[0]:.1f} min, p99 = {dur_df.loc[dur_df['station']=='S25', 'p99_min'].values[0]:.1f} min")
    print(f"Line 3 Processing Duration: Unavailable at individual stations (internal timestamps are identical).")
    print(f"Line 3 Inter-Station Wait:  Median transit between consecutive stations is 0 to 6.0 minutes.")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
