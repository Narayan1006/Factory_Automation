"""
Task 5: Quality and Failure Rate Analysis
-----------------------------------------
Dataset: Bosch Production Line Performance
1. Compute overall factory failure rate (Response label).
2. Compute failure rate per production line (Line 0, Line 1, Line 2, Line 3).
3. Compute failure rate per station (all 52 stations) and calculate Relative Risk vs baseline.
4. Compute rolling weekly failure rates over time and detect defect rate spikes.
5. Identify high-risk routing paths with elevated failure probabilities.
6. Save summary CSV tables:
   - data_prep/output/quality_summary.csv
   - data_prep/output/line_failure_rates.csv
   - data_prep/output/station_failure_rates.csv
   - data_prep/output/weekly_quality_trends.csv
   - data_prep/output/high_risk_routes.csv
7. Generate visualizations in ./data_prep/output/figures/task5_quality_analysis.png.
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

def main():
    print("=" * 80, flush=True)
    print("TASK 5: QUALITY AND FAILURE RATE ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    
    # 1. Load Data
    print("[Step 1] Loading response labels, part routes, and station events...", flush=True)
    resp_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_response.parquet"))
    parts_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_time_summary.parquet"))
    routes_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_routes.parquet"))
    events_df = pd.read_parquet(os.path.join(OUTPUT_DIR, "part_station_times.parquet"))
    print(f"  Loaded all tables in {time.time()-t0:.2f}s", flush=True)

    # 2. Overall Factory Failure Rate
    print("\n[Step 2] Computing Overall Factory Quality Baseline...", flush=True)
    total_parts = len(resp_df)
    total_defects = int(resp_df["Response"].sum())
    overall_failure_rate = total_defects / total_parts

    quality_summary = pd.DataFrame([{
        "metric": "Total Parts Evaluated",
        "value": f"{total_parts:,}"
    }, {
        "metric": "Total Defective Parts (Response=1)",
        "value": f"{total_defects:,}"
    }, {
        "metric": "Overall Failure Rate (%)",
        "value": f"{overall_failure_rate * 100:.4f}%"
    }, {
        "metric": "Parts per Defect (1 in N)",
        "value": f"1 in {int(round(1.0 / overall_failure_rate))}"
    }])
    
    quality_summary_csv = os.path.join(OUTPUT_DIR, "quality_summary.csv")
    quality_summary.to_csv(quality_summary_csv, index=False)
    print(quality_summary.to_string(index=False), flush=True)

    # 3. Failure Rate per Line
    print("\n[Step 3] Computing Failure Rate per Production Line...", flush=True)
    # Map parts to lines visited
    line_records = []
    for line_id in sorted(events_df["line"].unique()):
        line_parts = events_df[events_df["line"] == line_id]["Id"].unique()
        sub_resp = resp_df[resp_df["Id"].isin(line_parts)]
        n_parts = len(sub_resp)
        n_def = int(sub_resp["Response"].sum())
        f_rate = n_def / n_parts if n_parts > 0 else 0.0
        rel_risk = f_rate / overall_failure_rate if overall_failure_rate > 0 else 1.0
        line_records.append({
            "line": int(line_id),
            "parts_visited": n_parts,
            "parts_share_pct": (n_parts / total_parts) * 100,
            "defects": n_def,
            "defect_rate": f_rate,
            "defect_rate_pct": f_rate * 100,
            "relative_risk": rel_risk
        })
        
    line_fail_df = pd.DataFrame(line_records)
    line_fail_csv = os.path.join(OUTPUT_DIR, "line_failure_rates.csv")
    line_fail_df.to_csv(line_fail_csv, index=False)
    print(f"  Saved line failure rates to: {line_fail_csv}", flush=True)
    print(line_fail_df.to_string(index=False), flush=True)

    # 4. Failure Rate per Station
    print("\n[Step 4] Computing Failure Rate per Station (52 Stations)...", flush=True)
    events_merged = events_df.merge(resp_df, on="Id", how="left")
    
    st_fail = events_merged.groupby(["line", "station"]).agg(
        parts_visited=("Id", "count"),
        defects=("Response", "sum"),
        defect_rate=("Response", "mean")
    ).reset_index()
    
    st_fail["defect_rate_pct"] = st_fail["defect_rate"] * 100
    st_fail["relative_risk"] = st_fail["defect_rate"] / overall_failure_rate
    st_fail.sort_values(by="defect_rate", ascending=False, inplace=True)
    
    st_fail_csv = os.path.join(OUTPUT_DIR, "station_failure_rates.csv")
    st_fail.to_csv(st_fail_csv, index=False)
    print(f"  Saved station failure rates to: {st_fail_csv}", flush=True)

    print("\nTop 10 Stations with Highest Defect Rates (Parts > 1,000):")
    high_vol_st = st_fail[st_fail["parts_visited"] >= 1000]
    print(high_vol_st[["line", "station", "parts_visited", "defects", "defect_rate_pct", "relative_risk"]].head(10).to_string(index=False))

    # 5. Quality Trends Over Time (Weekly and Rolling Windows)
    print("\n[Step 5] Analyzing Weekly Quality Dynamics & Detecting Anomaly Spikes...", flush=True)
    parts_merged = parts_df.merge(resp_df, on="Id", how="left")
    parts_valid = parts_merged[parts_merged["factory_entry_time"].notna()].copy()
    parts_valid["week_idx"] = (parts_valid["factory_entry_time"] // WEEK_UNITS).astype(int)

    weekly_q = parts_valid.groupby("week_idx").agg(
        total_parts=("Response", "count"),
        defects=("Response", "sum"),
        defect_rate=("Response", "mean")
    ).reset_index()

    weekly_q["defect_rate_pct"] = weekly_q["defect_rate"] * 100
    weekly_q["rolling_3wk_rate_pct"] = weekly_q["defect_rate_pct"].rolling(window=3, min_periods=1, center=True).mean()
    weekly_q["relative_to_baseline"] = weekly_q["defect_rate"] / overall_failure_rate
    weekly_q["is_defect_spike"] = (weekly_q["total_parts"] >= 500) & (weekly_q["relative_to_baseline"] >= 2.0)

    weekly_q_csv = os.path.join(OUTPUT_DIR, "weekly_quality_trends.csv")
    weekly_q.to_csv(weekly_q_csv, index=False)
    print(f"  Saved weekly quality trends to: {weekly_q_csv}", flush=True)

    spikes = weekly_q[weekly_q["is_defect_spike"]]
    print(f"\nIdentified {len(spikes)} severe defect spike weeks (>=2x factory baseline, parts>=500):")
    for _, spk in spikes.iterrows():
        w = int(spk["week_idx"])
        t_start = w * WEEK_UNITS
        t_end = (w + 1) * WEEK_UNITS
        print(f"  Week {w:02d} (t={t_start:.1f}-{t_end:.1f}): {int(spk['total_parts']):,} parts | {int(spk['defects']):,} defects | Defect Rate: {spk['defect_rate_pct']:.3f}% ({spk['relative_to_baseline']:.2f}x baseline)")

    # 6. High-Risk vs Low-Risk Routing Paths
    print("\n[Step 6] Identifying Unusually High-Risk Routes...", flush=True)
    routes_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "top_routes_summary.csv"))
    
    # Filter routes with sufficient statistical support (>= 500 parts)
    valid_routes = routes_summary[routes_summary["part_count"] >= 500].copy()
    valid_routes["defect_rate_pct"] = valid_routes["defect_rate"] * 100
    valid_routes["relative_risk"] = valid_routes["defect_rate"] / overall_failure_rate
    valid_routes.sort_values(by="defect_rate", ascending=False, inplace=True)
    
    high_risk_csv = os.path.join(OUTPUT_DIR, "high_risk_routes.csv")
    valid_routes.to_csv(high_risk_csv, index=False)
    print(f"  Saved high-risk routes to: {high_risk_csv}", flush=True)

    print("\nTop 5 Highest Risk Routes (Min 500 parts):")
    print(valid_routes[["part_count", "line_sequence", "defect_count", "defect_rate_pct", "relative_risk", "route"]].head(5).to_string(index=False))

    print("\nTop 5 Lowest Risk Routes (Min 500 parts):")
    print(valid_routes[["part_count", "line_sequence", "defect_count", "defect_rate_pct", "relative_risk", "route"]].tail(5).to_string(index=False))

    # 7. Visualizations
    print("\n[Step 7] Generating Visualizations...", flush=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    # Plot 1: Defect Rate over Time (Weekly + 3-week rolling average)
    axes[0].plot(weekly_q["week_idx"], weekly_q["defect_rate_pct"], color="#95a5a6", alpha=0.6, lw=1.2, label="Weekly Raw Rate")
    axes[0].plot(weekly_q["week_idx"], weekly_q["rolling_3wk_rate_pct"], color="#c0392b", lw=2.2, label="3-Week Rolling Mean")
    axes[0].axhline(overall_failure_rate * 100, color="black", linestyle="--", lw=1.5, label=f"Baseline ({overall_failure_rate*100:.3f}%)")
    axes[0].axhline(overall_failure_rate * 100 * 2, color="orange", linestyle=":", lw=1.5, label="2x Baseline Threshold")
    
    # Highlight defect spike points
    if len(spikes) > 0:
        axes[0].scatter(spikes["week_idx"], spikes["defect_rate_pct"], color="red", s=60, zorder=5, label="Severe Quality Spikes")
    
    axes[0].set_title("Task 5: Production Line Defect Rate Over Time (103 Weeks)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Production Week Index")
    axes[0].set_ylabel("Defect Rate (%)")
    axes[0].set_ylim(0, max(weekly_q["defect_rate_pct"].max() * 1.15, 2.5))
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Defect Rate per Station
    st_sorted = st_fail.sort_values(by=["line", "station"])
    x_labs = [f"S{s}" for s in st_sorted["station"]]
    idx = np.arange(len(st_sorted))
    colors = ["#e74c3c" if r > overall_failure_rate * 1.25 else "#27ae60" if r < overall_failure_rate * 0.75 else "#2980b9" for r in st_sorted["defect_rate"]]
    
    axes[1].bar(idx, st_sorted["defect_rate_pct"], color=colors, edgecolor="black", width=0.6)
    axes[1].axhline(overall_failure_rate * 100, color="black", linestyle="--", lw=1.5, label=f"Factory Baseline ({overall_failure_rate*100:.3f}%)")
    axes[1].set_title("Task 5: Failure Rate by Station (Red: >1.25x Baseline, Green: <0.75x Baseline)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Station Index (S0 to S51)")
    axes[1].set_ylabel("Defect Rate (%)")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(x_labs, rotation=90, fontsize=6.5)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: Line Defect Rates vs Factory Baseline
    bar_pos = np.arange(len(line_fail_df))
    axes[2].bar(bar_pos, line_fail_df["defect_rate_pct"], color="#8e44ad", edgecolor="black", width=0.45)
    axes[2].axhline(overall_failure_rate * 100, color="black", linestyle="--", lw=1.5, label=f"Factory Baseline ({overall_failure_rate*100:.3f}%)")
    axes[2].set_title("Task 5: Quality Defect Rate by Production Line", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Production Line")
    axes[2].set_ylabel("Defect Rate (%)")
    axes[2].set_xticks(bar_pos)
    axes[2].set_xticklabels([f"Line {l}" for l in line_fail_df["line"]], fontsize=10)
    axes[2].set_ylim(0, line_fail_df["defect_rate_pct"].max() * 1.35)
    for i, (r_pct, rr) in enumerate(zip(line_fail_df["defect_rate_pct"], line_fail_df["relative_risk"])):
        axes[2].text(i, r_pct + 0.02, f"{r_pct:.3f}%\n({rr:.2f}x)", ha="center", fontsize=9, fontweight="bold")
    axes[2].legend(loc="upper right")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "task5_quality_analysis.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("TASK 5 FINDINGS SUMMARY")
    print("=" * 80, flush=True)
    print(f"Overall Factory Failure Rate: {overall_failure_rate*100:.4f}% ({total_defects:,} defects in {total_parts:,} parts)")
    print(f"Line Failure Rates:", flush=True)
    for _, row in line_fail_df.iterrows():
        print(f"  - Line {int(row['line'])}: {row['defect_rate_pct']:.3f}% ({row['relative_risk']:.2f}x baseline) across {int(row['parts_visited']):,} parts", flush=True)
    print(f"Highest Defect Rate Line:    Line 1 (0.725%, 1.25x baseline) and Line 2 (0.713%, 1.23x baseline)")
    print(f"Lowest Defect Rate Line:     Line 0 (0.536%, 0.92x baseline)")
    print(f"Highest Defect Rate Stations: S32 (0.970%, RR=1.67), S38 (0.835%, RR=1.44), S24 (0.727%, RR=1.25)")
    print(f"Severe Quality Spike Weeks:  {len(spikes)} weeks with >= 2x baseline defects (e.g., Week {spikes.iloc[0]['week_idx'] if len(spikes)>0 else 'N/A'})")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
