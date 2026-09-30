"""
Task 3: Production Line Routing Analysis
----------------------------------------
Dataset: Bosch Production Line Performance
1. For each part, compute the chronologically ordered sequence of visited stations.
2. Determine which production lines each part traversed.
3. Compute total number of distinct routes and rank the most common routes.
4. Merge quality failure labels (Response) to compute failure rates per route.
5. Save intermediate parquet and summary tables:
   - data_prep/output/part_routes.parquet
   - data_prep/output/top_routes_summary.csv
   - data_prep/output/line_combinations_summary.csv
6. Generate visualizations in ./data_prep/output/figures/task3_routes_analysis.png.
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

def main():
    print("=" * 80, flush=True)
    print("TASK 3: PART ROUTING ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    events_path = os.path.join(OUTPUT_DIR, "part_station_times.parquet")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing {events_path}. Run task2_time.py first!")

    print("[Step 1] Loading part station events...", flush=True)
    df = pd.read_parquet(events_path)
    print(f"  Loaded {len(df):,} station events in {time.time()-t0:.2f}s", flush=True)

    # Sort strictly by Id, entry_time, station
    print("[Step 2] Sorting chronologically per part...", flush=True)
    t1 = time.time()
    df_sorted = df.sort_values(by=["Id", "entry_time", "station"])
    print(f"  Sorted in {time.time()-t1:.2f}s", flush=True)

    # Group by Id to form ordered station sequences
    print("[Step 3] Constructing ordered routing sequences...", flush=True)
    t2 = time.time()
    
    # Fast grouping
    grouped = df_sorted.groupby("Id")
    part_routes = grouped["station"].apply(lambda s: "->".join(["S" + str(x) for x in s])).reset_index(name="route")
    part_lengths = grouped["station"].count().reset_index(name="num_stations")
    part_lines = grouped["line"].apply(lambda l: "->".join(["L" + str(x) for x in sorted(set(l))])).reset_index(name="line_sequence")
    
    routes_df = part_routes.merge(part_lengths, on="Id").merge(part_lines, on="Id")
    
    # Merge with part_response if available
    resp_path = os.path.join(OUTPUT_DIR, "part_response.parquet")
    if os.path.exists(resp_path):
        resp_df = pd.read_parquet(resp_path)
        routes_df = routes_df.merge(resp_df, on="Id", how="left")
        print("  Merged quality 'Response' labels.", flush=True)
    else:
        routes_df["Response"] = 0

    print(f"  Routes constructed for {len(routes_df):,} parts in {time.time()-t2:.2f}s", flush=True)

    # Save complete part routes table
    part_routes_parquet = os.path.join(OUTPUT_DIR, "part_routes.parquet")
    routes_df.to_parquet(part_routes_parquet)
    print(f"  Saved part routes to: {part_routes_parquet}", flush=True)

    # Step 4: Route Analysis
    print("\n[Step 4] Computing Distinct Routes and Rankings...", flush=True)
    route_stats = routes_df.groupby(["route", "line_sequence"]).agg(
        part_count=("Id", "count"),
        num_stations=("num_stations", "first"),
        defect_count=("Response", "sum"),
        defect_rate=("Response", "mean")
    ).reset_index()
    
    total_parts = len(routes_df)
    route_stats["volume_pct"] = (route_stats["part_count"] / total_parts) * 100
    route_stats.sort_values(by="part_count", ascending=False, inplace=True)
    
    # Save Top Routes summary
    top_routes_csv = os.path.join(OUTPUT_DIR, "top_routes_summary.csv")
    route_stats.to_csv(top_routes_csv, index=False)
    print(f"  Total distinct routes identified: {len(route_stats):,}")
    print(f"  Saved top routes to: {top_routes_csv}", flush=True)

    print("\nTop 15 Most Common Routes:")
    print(route_stats[["part_count", "volume_pct", "num_stations", "line_sequence", "defect_rate", "route"]].head(15).to_string(index=False))

    # Step 5: Line Combinations Breakdown
    print("\n[Step 5] Line Combinations Breakdown...", flush=True)
    line_stats = routes_df.groupby("line_sequence").agg(
        part_count=("Id", "count"),
        avg_stations=("num_stations", "mean"),
        defect_count=("Response", "sum"),
        defect_rate=("Response", "mean")
    ).reset_index()
    line_stats["volume_pct"] = (line_stats["part_count"] / total_parts) * 100
    line_stats.sort_values(by="part_count", ascending=False, inplace=True)
    
    line_stats_csv = os.path.join(OUTPUT_DIR, "line_combinations_summary.csv")
    line_stats.to_csv(line_stats_csv, index=False)
    print(f"  Saved line combinations to: {line_stats_csv}", flush=True)
    print(line_stats.to_string(index=False), flush=True)

    # Step 6: Visualizations
    print("\n[Step 6] Generating Visualizations...", flush=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Plot 1: Top 15 Most Common Routes
    top15 = route_stats.head(15).iloc[::-1] # reverse for horizontal bar
    axes[0].barh(top15["route"], top15["part_count"], color="#2c3e50", edgecolor="black")
    axes[0].set_title("Task 3: Top 15 Most Common Production Routes by Part Volume", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Parts")
    axes[0].set_ylabel("Station Sequence")
    for i, (cnt, pct) in enumerate(zip(top15["part_count"], top15["volume_pct"])):
        axes[0].text(cnt + 500, i, f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=8)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Plot 2: Line Combinations Share
    axes[1].bar(line_stats["line_sequence"], line_stats["part_count"], color="#3498db", edgecolor="black", width=0.5)
    axes[1].set_title("Task 3: Volume of Parts by Line Traversal Sequence", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Production Line Traversal Path")
    axes[1].set_ylabel("Total Parts")
    for i, (cnt, pct) in enumerate(zip(line_stats["part_count"], line_stats["volume_pct"])):
        axes[1].text(i, cnt + 10000, f"{cnt:,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: Distribution of Route Lengths (Number of Stations)
    length_counts = routes_df["num_stations"].value_counts().sort_index()
    axes[2].bar(length_counts.index, length_counts.values, color="#e67e22", edgecolor="black", width=0.7)
    axes[2].set_title("Task 3: Distribution of Stations Visited per Part (Route Length)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Number of Stations Visited")
    axes[2].set_ylabel("Number of Parts")
    axes[2].set_xticks(length_counts.index)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "task3_routes_analysis.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved figure: {fig_path}", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("TASK 3 FINDINGS SUMMARY")
    print("=" * 80, flush=True)
    print(f"Total Parts Routed:           {total_parts:,}", flush=True)
    print(f"Total Distinct Routes:        {len(route_stats):,}", flush=True)
    print(f"Top 10 Routes Account For:    {route_stats.head(10)['volume_pct'].sum():.2f}% of all parts", flush=True)
    print(f"Top 100 Routes Account For:   {route_stats.head(100)['volume_pct'].sum():.2f}% of all parts", flush=True)
    print("Primary Line Sequences:", flush=True)
    for _, row in line_stats.iterrows():
        print(f"  - {row['line_sequence']:<12}: {row['part_count']:>8,} parts ({row['volume_pct']:>5.2f}%) | Defect Rate: {row['defect_rate']*100:.3f}% | Avg Stations: {row['avg_stations']:.1f}", flush=True)
    print(f"Most Frequent Route:          {route_stats.iloc[0]['route']} ({route_stats.iloc[0]['part_count']:,} parts, {route_stats.iloc[0]['volume_pct']:.2f}%)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
