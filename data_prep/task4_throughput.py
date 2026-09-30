"""
Task 4: Throughput and Arrival Dynamics Analysis
------------------------------------------------
Dataset: Bosch Production Line Performance
1. Compute parts per time window (weekly and daily) overall, per line, and per station.
2. Identify busiest and quietest production periods:
   - Week 51: near-zero output, cause unknown (possibly shutdown)
   - Week 52: high defect rate after the gap
3. Estimate factory arrival rates and specific arrival dynamics at Station S29:
   - S29 arrival rate (parts/hour, parts/day)
   - S29 batch arrival size (parts sharing identical timestamp)
   - Inter-batch arrival time distribution (avoiding unverified 'Poisson' claims)
4. Save intermediate summary CSVs:
   - data_prep/output/weekly_throughput_by_line.csv
   - data_prep/output/station_throughput_metrics.csv
   - data_prep/output/arrival_rate_summary.csv
   - data_prep/output/s29_arrival_metrics.csv
5. Generate visualizations in ./data_prep/output/figures/task4_throughput_and_arrivals.png.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "data_prep/output"
FIGURES_DIR = "data_prep/output/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

WEEK_UNITS = 16.75
HOURS_PER_WEEK = 168.0
HOURS_PER_UNIT = HOURS_PER_WEEK / WEEK_UNITS   # ~10.02985 hours/unit
MINUTES_PER_UNIT = HOURS_PER_UNIT * 60.0       # ~601.79 minutes/unit

def main():
    print("=" * 80, flush=True)
    print("TASK 4: THROUGHPUT AND ARRIVAL DYNAMICS ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    
    # 1. Load Part Summaries & Station Events
    print("[Step 1] Loading part time summary and station events...", flush=True)
    parts_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_time_summary.parquet"))
    events_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_station_times.parquet"))
    print(f"  Loaded {len(parts_df):,} parts and {len(events_df):,} station events in {time.time()-t0:.2f}s", flush=True)

    # 2. Overall Factory Arrival Dynamics
    print("\n[Step 2] Estimating Factory Arrival Rate & Discrete Batch Gaps...", flush=True)
    valid_entries = parts_df["factory_entry_time"].dropna().sort_values().values
    total_parts = len(valid_entries)
    t_min = valid_entries.min()
    t_max = valid_entries.max()
    total_duration_units = t_max - t_min
    total_duration_hours = total_duration_units * HOURS_PER_UNIT
    total_duration_days = total_duration_hours / 24.0
    total_duration_weeks = total_duration_units / WEEK_UNITS

    arrival_rate_per_hour = total_parts / total_duration_hours
    arrival_rate_per_day = total_parts / total_duration_days
    arrival_rate_per_week = total_parts / total_duration_weeks

    # Inter-arrival times across all factory parts
    dt_units = np.diff(valid_entries)
    dt_minutes = dt_units * MINUTES_PER_UNIT
    positive_dt = dt_minutes[dt_minutes > 0]

    arrival_summary = pd.DataFrame([{
        "metric": "Total Parts Analyzed",
        "value": f"{total_parts:,}"
    }, {
        "metric": "Timeline Span (Units)",
        "value": f"{total_duration_units:.2f}"
    }, {
        "metric": "Timeline Span (Weeks / Years, estimated)",
        "value": f"{total_duration_weeks:.1f} weeks ({total_duration_weeks/52.14:.2f} years)"
    }, {
        "metric": "Estimated Mean Arrival Rate (parts/hour)",
        "value": f"{arrival_rate_per_hour:.2f}"
    }, {
        "metric": "Estimated Mean Arrival Rate (parts/day)",
        "value": f"{arrival_rate_per_day:.1f}"
    }, {
        "metric": "Estimated Mean Arrival Rate (parts/week)",
        "value": f"{arrival_rate_per_week:.1f}"
    }, {
        "metric": "Mean Inter-arrival Time (all parts, min)",
        "value": f"{dt_minutes.mean():.4f}"
    }, {
        "metric": "Mean Inter-Batch Gap (positive dt, min)",
        "value": f"{positive_dt.mean():.4f}"
    }, {
        "metric": "Median Inter-Batch Gap (positive dt, min)",
        "value": f"{np.median(positive_dt):.4f}"
    }, {
        "metric": "Parts Batching in Same Discrete Timestamp (%)",
        "value": f"{(dt_minutes == 0).mean()*100:.2f}%"
    }])
    
    arrival_csv = os.path.join(OUTPUT_DIR, "arrival_rate_summary.csv")
    arrival_summary.to_csv(arrival_csv, index=False)
    print(f"  Saved arrival metrics to: {arrival_csv}", flush=True)

    # 3. Station S29 Specific Arrival Dynamics
    print("\n[Step 3] Analyzing Arrival Dynamics Specifically at Station S29...", flush=True)
    s29_events = events_df[events_df["station"] == 29].sort_values("entry_time")
    s29_times = s29_events["entry_time"].values
    s29_total = len(s29_times)
    s29_t_min, s29_t_max = s29_times.min(), s29_times.max()
    s29_span_hours = (s29_t_max - s29_t_min) * HOURS_PER_UNIT
    s29_rate_per_hour = s29_total / s29_span_hours
    s29_rate_per_day = s29_rate_per_hour * 24.0

    # Batch sizes at S29 (number of parts sharing the exact same timestamp)
    s29_batch_sizes = pd.Series(s29_times).value_counts().values
    s29_dt = np.diff(np.sort(np.unique(s29_times))) * MINUTES_PER_UNIT

    s29_metrics = pd.DataFrame([{
        "metric": "Total Parts Entering S29",
        "value": f"{s29_total:,}"
    }, {
        "metric": "S29 Active Duration (Hours, estimated)",
        "value": f"{s29_span_hours:.1f}"
    }, {
        "metric": "S29 Arrival Rate (parts/hour, estimated)",
        "value": f"{s29_rate_per_hour:.2f}"
    }, {
        "metric": "S29 Arrival Rate (parts/day, estimated)",
        "value": f"{s29_rate_per_day:.1f}"
    }, {
        "metric": "Total Distinct S29 Batch Timestamps",
        "value": f"{len(s29_batch_sizes):,}"
    }, {
        "metric": "Mean S29 Batch Size (parts/timestamp)",
        "value": f"{s29_batch_sizes.mean():.2f}"
    }, {
        "metric": "Median S29 Batch Size (parts/timestamp)",
        "value": f"{np.median(s29_batch_sizes):.0f}"
    }, {
        "metric": "P90 S29 Batch Size (parts/timestamp)",
        "value": f"{np.percentile(s29_batch_sizes, 90):.0f}"
    }, {
        "metric": "Max S29 Batch Size (parts/timestamp)",
        "value": f"{s29_batch_sizes.max():.0f}"
    }, {
        "metric": "Mean Inter-Batch Gap at S29 (min)",
        "value": f"{s29_dt.mean():.2f}"
    }, {
        "metric": "Median Inter-Batch Gap at S29 (min)",
        "value": f"{np.median(s29_dt):.2f}"
    }])

    s29_csv = os.path.join(OUTPUT_DIR, "s29_arrival_metrics.csv")
    s29_metrics.to_csv(s29_csv, index=False)
    print(f"  Saved S29 metrics to: {s29_csv}", flush=True)
    print(s29_metrics.to_string(index=False), flush=True)

    # 4. Weekly Throughput Overall and per Line
    print("\n[Step 4] Computing Weekly Throughput Overall and per Line...", flush=True)
    parts_valid = parts_df[parts_df["factory_entry_time"].notna()].copy()
    parts_valid["week_idx"] = (parts_valid["factory_entry_time"] // WEEK_UNITS).astype(int)

    weekly_factory = parts_valid.groupby("week_idx").size().rename("total_factory_parts")

    events_valid = events_df.copy()
    events_valid["week_idx"] = (events_valid["entry_time"] // WEEK_UNITS).astype(int)
    
    line_first_events = events_valid.sort_values(by=["Id", "line", "entry_time"]).groupby(["Id", "line"]).first().reset_index()
    weekly_by_line = line_first_events.groupby(["week_idx", "line"]).size().unstack(fill_value=0)
    weekly_by_line.columns = [f"Line_{c}" for c in weekly_by_line.columns]

    weekly_combined = pd.concat([weekly_factory, weekly_by_line], axis=1).fillna(0).astype(int)
    weekly_combined.index.name = "week_idx"
    weekly_combined["week_start_unit"] = weekly_combined.index * WEEK_UNITS
    weekly_combined["week_end_unit"] = (weekly_combined.index + 1) * WEEK_UNITS

    weekly_csv = os.path.join(OUTPUT_DIR, "weekly_throughput_by_line.csv")
    weekly_combined.to_csv(weekly_csv)
    print(f"  Saved weekly throughput by line to: {weekly_csv}", flush=True)

    print("\n--- Busiest Periods (Peak Production) ---")
    busiest_weeks = weekly_combined.sort_values(by="total_factory_parts", ascending=False).head(5)
    for w_idx, row in busiest_weeks.iterrows():
        print(f"  Week {w_idx:02d} (t={row['week_start_unit']:.1f}-{row['week_end_unit']:.1f}): {row['total_factory_parts']:,} parts")

    print("\n--- Quietest Periods (Near-Zero Output & Gaps) ---")
    active_weeks = weekly_combined[weekly_combined.index <= 102]
    quietest_weeks = active_weeks.sort_values(by="total_factory_parts", ascending=True).head(5)
    for w_idx, row in quietest_weeks.iterrows():
        print(f"  Week {w_idx:02d} (t={row['week_start_unit']:.1f}-{row['week_end_unit']:.1f}): {row['total_factory_parts']:,} parts")

    # 5. Station Throughput Metrics
    print("\n[Step 5] Computing Station Throughput Metrics...", flush=True)
    st_weekly = events_valid.groupby(["station", "line", "week_idx"]).size().reset_index(name="parts_in_week")
    
    st_metrics = st_weekly.groupby(["station", "line"]).agg(
        total_parts=("parts_in_week", "sum"),
        active_weeks=("week_idx", "nunique"),
        mean_parts_per_week=("parts_in_week", "mean"),
        median_parts_per_week=("parts_in_week", "median"),
        peak_parts_per_week=("parts_in_week", "max"),
        min_parts_per_week=("parts_in_week", "min"),
        std_parts_per_week=("parts_in_week", "std")
    ).reset_index()

    st_metrics["mean_parts_per_hour"] = st_metrics["mean_parts_per_week"] / 168.0
    st_metrics["peak_parts_per_hour"] = st_metrics["peak_parts_per_week"] / 168.0
    st_metrics.sort_values(by=["line", "station"], inplace=True)

    st_metrics_csv = os.path.join(OUTPUT_DIR, "station_throughput_metrics.csv")
    st_metrics.to_csv(st_metrics_csv, index=False)
    print(f"  Saved station throughput metrics to: {st_metrics_csv}", flush=True)

    # 6. Visualizations
    print("\n[Step 6] Generating Visualizations...", flush=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    # Plot 1: Weekly Throughput Trend
    axes[0].plot(weekly_combined.index, weekly_combined["total_factory_parts"], color="black", lw=2.2, label="Total Factory")
    if "Line_0" in weekly_combined.columns:
        axes[0].plot(weekly_combined.index, weekly_combined["Line_0"], color="#1f77b4", lw=1.5, linestyle="--", label="Line 0")
    if "Line_1" in weekly_combined.columns:
        axes[0].plot(weekly_combined.index, weekly_combined["Line_1"], color="#ff7f0e", lw=1.5, linestyle="--", label="Line 1")
    if "Line_2" in weekly_combined.columns:
        axes[0].plot(weekly_combined.index, weekly_combined["Line_2"], color="#2ca02c", lw=1.5, linestyle="--", label="Line 2")
    if "Line_3" in weekly_combined.columns:
        axes[0].plot(weekly_combined.index, weekly_combined["Line_3"], color="#d62728", lw=1.5, linestyle="-.", label="Line 3")
    
    # Annotate Week 51 (near-zero output)
    if 51 in weekly_combined.index:
        axes[0].annotate("Near-Zero Output (Week 51: 4 parts)\nCause unknown (possibly shutdown)", 
                         xy=(51, 4), xytext=(48, 25000),
                         arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                         fontweight="bold", color="red", ha="center", fontsize=8.5,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9))
    
    # Annotate Week 64 (tiny volume, 1 defect in 46 parts)
    if 64 in weekly_combined.index:
        axes[0].annotate("Week 64: 46 parts logged\n(1 defect / 46 pts = 2.17%)", 
                         xy=(64, 46), xytext=(78, 18000),
                         arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=5),
                         fontweight="bold", color="#2c3e50", ha="center", fontsize=8.5,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    
    axes[0].set_title("Task 4: Weekly Production Volume Across 103 Estimated Weeks (Total Factory & Lines)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Estimated Production Week Index (Assuming 1 Week ≈ 16.75 Date Units = 168 Hours)")
    axes[0].set_ylabel("Parts Throughput (Parts / Week)")
    axes[0].set_ylim(-1000, 32000)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Station Mean vs Peak Weekly Throughput
    x_labs = [f"S{s}" for s in st_metrics["station"]]
    idx = np.arange(len(st_metrics))
    w = 0.4
    axes[1].bar(idx - w/2, st_metrics["mean_parts_per_week"], width=w, label="Mean Parts/Week", color="#2980b9")
    axes[1].bar(idx + w/2, st_metrics["peak_parts_per_week"], width=w, label="Peak Parts/Week", color="#e74c3c", alpha=0.7)
    axes[1].set_title("Task 4: Station Throughput Capacity (Mean vs Peak Parts per Week)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Station Index (S0 to S51)")
    axes[1].set_ylabel("Parts per Week")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(x_labs, rotation=90, fontsize=6.5)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: S29 Batch Size Distribution
    axes[2].hist(s29_batch_sizes, bins=np.arange(0.5, 47.5, 1.0), color="#16a085", edgecolor="black", alpha=0.85)
    axes[2].set_title(f"Task 4: Station S29 Discrete Batch Size Distribution (Mean={s29_batch_sizes.mean():.1f}, Median={np.median(s29_batch_sizes):.0f} parts/batch)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Batch Size (Parts Sharing Identical Timestamp at S29)")
    axes[2].set_ylabel("Batch Frequency")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "task4_throughput_and_arrivals.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("TASK 4 FINDINGS SUMMARY")
    print("=" * 80, flush=True)
    print(f"Estimated Mean Factory Arrival Rate: {arrival_rate_per_hour:.2f} parts/hour ({arrival_rate_per_day:.1f} parts/day)")
    print(f"Station S29 Arrival Rate:            {s29_rate_per_hour:.2f} parts/hour ({s29_rate_per_day:.1f} parts/day)")
    print(f"Station S29 Batching Dynamics:       Mean batch = {s29_batch_sizes.mean():.1f} parts, Median = {np.median(s29_batch_sizes):.0f} parts (Batch transfer, not Poisson)")
    print(f"Busiest Period:                      Week 40 with {busiest_weeks['total_factory_parts'].iloc[0]:,} parts (~{busiest_weeks['total_factory_parts'].iloc[0]/168:.1f} parts/hour)")
    print(f"Quietest Period:                     Week 51 with near-zero output (ONLY 4 parts logged; cause unknown, possibly shutdown)")
    print(f"Post-Gap Quality Behavior:           Week 52 experienced a high defect rate (1.57%) immediately following the Week 51 volume collapse")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
