import os
import sys
import math
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"c:\projects\HCL_Projects\PROJECT"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_prep", "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=== STEP 1: FILTERING FEATURES ===")
rec_feats_df = pd.read_csv(os.path.join(OUTPUT_DIR, "recommended_subset_features.csv"))
# Also add S35 features
# Check what S35 features exist
s35_feats = [
    {'line': 3, 'station': 'S35', 'col_name': 'L3_S35_F3884', 'feature_id': 3884, 'sample_non_null': 550000, 'missing_pct_sample': 53.4, 'mean': 0.0, 'std': 0.00005, 'variance': 0.0, 'min': -0.1, 'max': 0.1},
    {'line': 3, 'station': 'S35', 'col_name': 'L3_S35_F3889', 'feature_id': 3889, 'sample_non_null': 550000, 'missing_pct_sample': 53.4, 'mean': 0.0, 'std': 0.05528, 'variance': 0.00305, 'min': -0.5, 'max': 0.5},
    {'line': 3, 'station': 'S35', 'col_name': 'L3_S35_F3894', 'feature_id': 3894, 'sample_non_null': 550000, 'missing_pct_sample': 53.4, 'mean': 0.0, 'std': 0.07830, 'variance': 0.00613, 'min': -0.5, 'max': 0.5},
    {'line': 3, 'station': 'S35', 'col_name': 'L3_S35_F3896', 'feature_id': 3896, 'sample_non_null': 550000, 'missing_pct_sample': 53.4, 'mean': 0.0, 'std': 0.14053, 'variance': 0.01975, 'min': -0.6, 'max': 0.6}
]
df_s35 = pd.DataFrame(s35_feats)

combined_feats = pd.concat([rec_feats_df, df_s35], ignore_index=True)

# Low quality list identified:
# std < 0.01 or unique < 10
low_quality_cols = {
    'L3_S29_F3360',  # 4 unique
    'L3_S30_F3779',  # 2 unique
    'L3_S30_F3739',  # std 0.0028
    'L3_S30_F3734',  # std 0.0015
    'L3_S30_F3789',  # 2 unique
    'L3_S30_F3814',  # 2 unique
    'L3_S30_F3824',  # 2 unique
    'L3_S30_F3729',  # std 0.0001
    'L3_S30_F3724',  # std 0.0013
    'L3_S30_F3494',  # std 0.0087
    'L3_S30_F3499',  # std 0.0087
    'L3_S33_F3867',  # std 0.0008
    'L3_S33_F3869',  # std 0.0031
    'L3_S33_F3871',  # std 0.0010
    'L3_S33_F3873',  # std 0.0014
    'L3_S34_F3876',  # std 0.0013
    'L3_S34_F3878',  # std 0.0009
    'L3_S34_F3880',  # std 0.0009
    'L3_S34_F3882',  # std 0.0022
    'L3_S35_F3884',  # std 0.00005, 2 unique
    'L3_S35_F3894',  # 7 unique
    'L3_S36_F3938',  # std 0.0033
    'L3_S36_F3934',  # std 0.0020
    'L3_S36_F3930',  # std 0.0010
    'L3_S36_F3926',  # std 0.0017
    'L3_S37_F3944',  # std 0.0001
    'L3_S37_F3946',  # std 0.0009
    'L3_S37_F3948',  # std 0.0012
    'L3_S37_F3950',  # std 0.0022
}

filtered_feats = combined_feats[~combined_feats['col_name'].isin(low_quality_cols)].copy()
print(f"Features remaining after filtering (std >= 0.01 & unique >= 10): {len(filtered_feats)}")
print(filtered_feats.groupby('station')['col_name'].count())
filtered_feats.to_csv(os.path.join(OUTPUT_DIR, "recommended_subset_features_filtered.csv"), index=False)

print("\n=== STEP 2: GROUPING ANOMALIES BY TIME INTO LINE-LEVEL EVENTS ===")
# Line 3 stations: S29, S30, S33, S34, S35, S36, S37
# Defect surges are line-level time events because parts traverse these stations within minutes.
# Load part station times and responses
part_station_times = pq.read_table(os.path.join(OUTPUT_DIR, "part_station_times.parquet")).to_pandas()
part_resp = pq.read_table(os.path.join(OUTPUT_DIR, "part_response.parquet")).to_pandas()

# Filter to Line 3 core stations
l3_stations = [29, 30, 31, 33, 34, 35, 36, 37]
l3_events = part_station_times[part_station_times['station'].isin(l3_stations)].copy()
l3_events = l3_events.merge(part_resp, on='Id', how='inner')

# Identify the main candidate time windows across Line 3:
# Event A: t ~ 497.0 - 500.0 (S33 & S34 defect surge)
# Event B: t ~ 736.0 - 739.0 (S29 defect surge)
# Event C: t ~ 1607.0 - 1610.0 (S29 defect surge)
# Event D: t ~ 371.0 - 378.0 (S29 / S37 defect surge)
# Event E: t ~ 366.0 - 369.0 (S29 measurement drift on F3339)
# Event F: t ~ 382.0 - 385.0 (S37 measurement drift on F3950)
# Event G: t ~ 0.0 - 3.0 (S29 measurement drift on F3357)
# Event H: t ~ 24.0 - 40.0 (S31 start-up transient drift on F3834)

# Let's compute exact metrics for line-level defect surge events
baseline_defect_rate = 6879 / 1183747  # 0.005811

defect_event_windows = [
    {"event_id": "EV-DEF-01", "name": "t~497 Defect Surge (S33/S34 Batch)", "t_start": 497.0, "t_end": 500.0, "stations": [33, 34]},
    {"event_id": "EV-DEF-02", "name": "t~1608 Defect Surge (S29 Shift)", "t_start": 1607.38, "t_end": 1609.77, "stations": [29]},
    {"event_id": "EV-DEF-03", "name": "t~737 Defect Surge (S29 High Volume)", "t_start": 736.38, "t_end": 738.77, "stations": [29]},
    {"event_id": "EV-DEF-04", "name": "t~372 Defect Surge (S29/S37 Sequence)", "t_start": 371.46, "t_end": 378.64, "stations": [29, 37]},
    {"event_id": "EV-DEF-05", "name": "t~372 24h Defect Peak (S29)", "t_start": 372.38, "t_end": 374.77, "stations": [29]}
]

event_records = []
for ev in defect_event_windows:
    t_s = ev['t_start']
    t_e = ev['t_end']
    st_list = ev['stations']
    sub = l3_events[(l3_events['station'].isin(st_list)) & (l3_events['entry_time'] >= t_s) & (l3_events['entry_time'] <= t_e)]
    
    distinct_parts = sub['Id'].nunique()
    failed_sub = sub[sub['Response'] == 1]
    distinct_failed = failed_sub['Id'].nunique()
    
    # Check overlap across stations if multiple stations
    if len(st_list) > 1:
        st_fails = [set(failed_sub[failed_sub['station'] == s]['Id'].unique()) for s in st_list]
        intersection = len(set.intersection(*st_fails)) if all(len(s) > 0 for s in st_fails) else 0
        overlap_note = f"Consecutive overlap: {intersection}/{distinct_failed} failed parts ({intersection/max(1,distinct_failed)*100:.1f}%)"
    else:
        overlap_note = "Single station window"
        
    obs_rate = (distinct_failed / distinct_parts) * 100 if distinct_parts > 0 else 0
    expected_failures = distinct_parts * baseline_defect_rate
    excess_failures = distinct_failed - expected_failures
    oe_ratio = distinct_failed / max(1e-6, expected_failures)
    
    # Binomial p-value approximation via normal approximation
    if distinct_parts > 0:
        p_base = baseline_defect_rate
        mean_k = distinct_parts * p_base
        std_k = math.sqrt(distinct_parts * p_base * (1 - p_base))
        z_stat = (distinct_failed - mean_k) / std_k
        p_val = 0.5 * math.erfc(z_stat / math.sqrt(2))
    else:
        p_val = 1.0
        
    st_names = ", ".join([f"S{s}" for s in st_list])
    event_records.append({
        "event_id": ev['event_id'],
        "event_type": "Defect Surge",
        "stations_involved": st_names,
        "t_start": t_s,
        "t_end": t_e,
        "duration_hours": (t_e - t_s) * 10.03,
        "total_distinct_parts": distinct_parts,
        "observed_failures": distinct_failed,
        "expected_failures": round(expected_failures, 2),
        "excess_failures": round(excess_failures, 2),
        "oe_ratio": round(oe_ratio, 2),
        "observed_defect_rate_pct": round(obs_rate, 2),
        "baseline_rate_pct": round(baseline_defect_rate * 100, 3),
        "p_value": p_val,
        "overlap_dedup_note": overlap_note
    })

# Add measurement drift events (excluding S37 F3950 which is filtered out)
drift_event_windows = [
    {"event_id": "EV-DRF-01", "name": "t~367 S29 Sensor Drift", "station": 29, "feat": "L3_S29_F3339", "t_start": 366.38, "t_end": 368.77, "z": 2.52},
    {"event_id": "EV-DRF-02", "name": "t~577 S29 Sensor Drift", "station": 29, "feat": "L3_S29_F3339", "t_start": 576.38, "t_end": 578.77, "z": -3.40},
    {"event_id": "EV-DRF-03", "name": "t~1.5 S29 Sensor Drift", "station": 29, "feat": "L3_S29_F3357", "t_start": 0.38, "t_end": 2.77, "z": 2.64},
    {"event_id": "EV-DRF-04", "name": "t~24-40 S31 Start-up Transient", "station": 31, "feat": "L3_S31_F3834", "t_start": 24.33, "t_end": 40.00, "z": 7.65}
]

for dev in drift_event_windows:
    t_s = dev['t_start']
    t_e = dev['t_end']
    st = dev['station']
    sub = l3_events[(l3_events['station'] == st) & (l3_events['entry_time'] >= t_s) & (l3_events['entry_time'] <= t_e)]
    parts_cnt = sub['Id'].nunique()
    z = dev['z']
    p_val = math.erfc(abs(z) / math.sqrt(2))
    event_records.append({
        "event_id": dev['event_id'],
        "event_type": "Measurement Drift",
        "stations_involved": f"S{st}",
        "t_start": t_s,
        "t_end": t_e,
        "duration_hours": (t_e - t_s) * 10.03,
        "total_distinct_parts": parts_cnt,
        "observed_failures": "N/A (Sensor Drift)",
        "expected_failures": "N/A",
        "excess_failures": "N/A",
        "oe_ratio": f"{abs(z):.2f} sigma",
        "observed_defect_rate_pct": f"Feature {dev['feat']}",
        "baseline_rate_pct": "mu=0, sigma=1",
        "p_value": p_val,
        "overlap_dedup_note": f"Z-score = {z:+.2f} sigma"
    })

df_events = pd.DataFrame(event_records)
defect_events_ranked = df_events[df_events['event_type'] == 'Defect Surge'].sort_values(by='oe_ratio', ascending=False)
drift_events_ranked = df_events[df_events['event_type'] == 'Measurement Drift'].sort_values(by='oe_ratio', ascending=False)

df_all_events_ranked = pd.concat([defect_events_ranked, drift_events_ranked], ignore_index=True)
df_all_events_ranked.to_csv(os.path.join(OUTPUT_DIR, "grouped_anomaly_events.csv"), index=False)
print("\nGrouped Anomaly Events (Ranked by Observed vs Expected):")
print(df_all_events_ranked[['event_id', 'event_type', 'stations_involved', 't_start', 't_end', 'total_distinct_parts', 'observed_failures', 'expected_failures', 'oe_ratio']])

print("\n=== STEP 3: VERIFY t~366-385 CO-OCCURRING SIGNALS (S29 DRIFT & DEFECT SURGE) ===")
# In t in [362, 388]:
chunk_size = 100000
seq_parts = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "train_numeric.csv"), usecols=['Id', 'L3_S29_F3339', 'Response'], chunksize=chunk_size):
    sub = chunk[chunk['Id'].isin(l3_events['Id'])]
    seq_parts.append(sub)
df_all_seq = pd.concat(seq_parts, ignore_index=True)

# Merge with S29 entry times
s29_times = l3_events[l3_events['station'] == 29][['Id', 'entry_time']].rename(columns={'entry_time': 't_s29'})
df_seq_merged = df_all_seq.merge(s29_times, on='Id', how='left')

# Filter to window t in [362, 388]
df_seq_window = df_seq_merged[(df_seq_merged['t_s29'] >= 362) & (df_seq_merged['t_s29'] <= 388)].copy()

mu_f3339, std_f3339 = -0.0003, 0.1213

# Create uniform grid of 0.5 units (~5 hours) from 362 to 388
all_bins = np.arange(362.0, 388.5, 0.5)
df_seq_window['t_bin'] = np.floor(df_seq_window['t_s29'] * 2) / 2

bin_summary = df_seq_window.groupby('t_bin').agg(
    parts_count=('Id', 'count'),
    defects=('Response', 'sum'),
    defect_rate=('Response', 'mean'),
    f3339_mean=('L3_S29_F3339', 'mean')
).reindex(all_bins).reset_index().rename(columns={'index': 't_bin'})

# Set NaN where parts_count == 0 to avoid interpolating over gaps
bin_summary['f3339_z'] = np.where(bin_summary['parts_count'] > 0, (bin_summary['f3339_mean'] - mu_f3339) / std_f3339, np.nan)
bin_summary['defect_rate_pct'] = np.where(bin_summary['parts_count'] > 0, bin_summary['defect_rate'] * 100, np.nan)
bin_summary['parts_count'] = bin_summary['parts_count'].fillna(0)

print("\nCo-occurring Signals Timeline in [364, 386]:")
print(bin_summary[(bin_summary['t_bin'] >= 364) & (bin_summary['t_bin'] <= 386)][['t_bin', 'parts_count', 'defects', 'defect_rate_pct', 'f3339_z']])

# Generate Dedicated Plot: task7b_sequence_366_385.png (Co-occurring Signals, NO S37 drift)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Subplot 1: S29 Sensor Drift + Part Counts
ax1_twin = ax1.twinx()
ax1_twin.bar(bin_summary['t_bin'], bin_summary['parts_count'], width=0.4, color='#bdc3c7', alpha=0.45, label='Parts Count / 5h Bin')
ax1_twin.set_ylabel('Parts Processed / Bin', color='#7f8c8d', fontsize=11)
ax1_twin.grid(False)

ax1.plot(bin_summary['t_bin'], bin_summary['f3339_z'], color='#1f77b4', marker='o', linewidth=2, label='S29: L3_S29_F3339 (Z-score, NaN on gaps)')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.6)
ax1.axhline(2.0, color='red', linestyle=':', label='+2σ Drift Threshold')
ax1.axvspan(366.38, 368.77, color='#1f77b4', alpha=0.15, label='S29 Drift Window (t=366.4-368.8)')
ax1.set_ylabel('S29 Sensor Drift (σ)', fontsize=11, color='#1f77b4')
ax1.set_title('Signal 1: Upstream Station S29 Telemetry Drift (Z-Score)', fontsize=12, fontweight='bold', loc='left')
# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
ax1.grid(True, alpha=0.3)

# Subplot 2: Factory Defect Rate + Part Counts
ax2_twin = ax2.twinx()
ax2_twin.bar(bin_summary['t_bin'], bin_summary['parts_count'], width=0.4, color='#bdc3c7', alpha=0.45, label='Parts Count / 5h Bin')
ax2_twin.set_ylabel('Parts Processed / Bin', color='#7f8c8d', fontsize=11)
ax2_twin.grid(False)

ax2.plot(bin_summary['t_bin'], bin_summary['defect_rate_pct'], color='#d62728', marker='s', linewidth=2, label='Observed Defect Rate (%)')
ax2.axhline(baseline_defect_rate * 100, color='black', linestyle='--', label=f'Factory Baseline ({baseline_defect_rate*100:.2f}%)')
ax2.axvspan(371.46, 378.64, color='#d62728', alpha=0.15, label='Defect Surge Window (t=371.5-378.6)')
ax2.set_ylabel('Defect Rate (%)', fontsize=11, color='#d62728')
ax2.set_xlabel('Timestamp (Relative Time Units; 1 unit ≈ 10.03 hours; bin = 0.5 units ≈ 5 hours)', fontsize=12)
ax2.set_title('Signal 2: Concurrent Quality Failure Surge (Elevated Response=1, Peaking at 5.04% at t=374.0)', fontsize=12, fontweight='bold', loc='left')
lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right', frameon=True)
ax2.grid(True, alpha=0.3)

plt.suptitle('Co-occurring Signals: Upstream Station S29 Sensor Drift (t~367) and Quality Failure Surge (t~372-378)', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "task7b_sequence_366_385.png"), dpi=300)
plt.close(fig)
print("Saved task7b_sequence_366_385.png")

print("\n=== STEP 4: LINE 3 TRANSIT TIMES OVER TIME (p90, p99, Zero-Delta Share) ===")
# Compute transit times per part: S29 -> S30, S30 -> S33, S33 -> S34, S34 -> S36, S36 -> S37
s29_times = l3_events[l3_events['station'] == 29][['Id', 'exit_time']].rename(columns={'exit_time': 't_s29_exit'})
s30_times = l3_events[l3_events['station'] == 30][['Id', 'entry_time', 'exit_time']].rename(columns={'entry_time': 't_s30_entry', 'exit_time': 't_s30_exit'})
s33_times = l3_events[l3_events['station'] == 33][['Id', 'entry_time', 'exit_time']].rename(columns={'entry_time': 't_s33_entry', 'exit_time': 't_s33_exit'})
s34_times = l3_events[l3_events['station'] == 34][['Id', 'entry_time', 'exit_time']].rename(columns={'entry_time': 't_s34_entry', 'exit_time': 't_s34_exit'})
s36_times = l3_events[l3_events['station'] == 36][['Id', 'entry_time', 'exit_time']].rename(columns={'entry_time': 't_s36_entry', 'exit_time': 't_s36_exit'})
s37_times = l3_events[l3_events['station'] == 37][['Id', 'entry_time']].rename(columns={'entry_time': 't_s37_entry'})

df_trans = s29_times.merge(s30_times, on='Id', how='inner')\
                    .merge(s33_times, on='Id', how='inner')\
                    .merge(s34_times, on='Id', how='inner')\
                    .merge(s36_times, on='Id', how='inner')\
                    .merge(s37_times, on='Id', how='inner')

unit_to_min = 10.02985 * 60.0
df_trans['delta_29_30'] = (df_trans['t_s30_entry'] - df_trans['t_s29_exit']) * unit_to_min
df_trans['delta_30_33'] = (df_trans['t_s33_entry'] - df_trans['t_s30_exit']) * unit_to_min
df_trans['delta_33_34'] = (df_trans['t_s34_entry'] - df_trans['t_s33_exit']) * unit_to_min
df_trans['delta_34_36'] = (df_trans['t_s36_entry'] - df_trans['t_s34_exit']) * unit_to_min
df_trans['delta_36_37'] = (df_trans['t_s37_entry'] - df_trans['t_s36_exit']) * unit_to_min
df_trans['week'] = (df_trans['t_s29_exit'] // 16.75).astype(int)

# Group by week and compute p90, p99, and zero_delta_pct
weekly_trans = df_trans.groupby('week').agg(
    p90_29_30=('delta_29_30', lambda x: np.percentile(x, 90)),
    p99_29_30=('delta_29_30', lambda x: np.percentile(x, 99)),
    zero_29_30=('delta_29_30', lambda x: (x == 0).mean() * 100),
    
    p90_30_33=('delta_30_33', lambda x: np.percentile(x, 90)),
    p99_30_33=('delta_30_33', lambda x: np.percentile(x, 99)),
    zero_30_33=('delta_30_33', lambda x: (x == 0).mean() * 100),
    
    p90_33_34=('delta_33_34', lambda x: np.percentile(x, 90)),
    p99_33_34=('delta_33_34', lambda x: np.percentile(x, 99)),
    zero_33_34=('delta_33_34', lambda x: (x == 0).mean() * 100),
    
    p90_34_36=('delta_34_36', lambda x: np.percentile(x, 90)),
    p99_34_36=('delta_34_36', lambda x: np.percentile(x, 99)),
    zero_34_36=('delta_34_36', lambda x: (x == 0).mean() * 100),
    
    p90_36_37=('delta_36_37', lambda x: np.percentile(x, 90)),
    p99_36_37=('delta_36_37', lambda x: np.percentile(x, 99)),
    zero_36_37=('delta_36_37', lambda x: (x == 0).mean() * 100)
).reset_index()

weekly_trans = weekly_trans[weekly_trans['week'] <= 102]

# Plot Line 3 Transit Times Over Time (3 Panels: p90, p99, zero-delta share)
fig, (ax_p90, ax_p99, ax_zero) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

# Panel 1: 90th Percentile Delay (p90)
ax_p90.plot(weekly_trans['week'], weekly_trans['p90_30_33'], color='#d62728', marker='o', markersize=3, label='S30->S33 (Buffer Delay)')
ax_p90.plot(weekly_trans['week'], weekly_trans['p90_29_30'], color='#1f77b4', marker='s', markersize=3, label='S29->S30')
ax_p90.plot(weekly_trans['week'], weekly_trans['p90_34_36'], color='#e67e22', marker='v', markersize=3, label='S34->S36')
ax_p90.plot(weekly_trans['week'], weekly_trans['p90_33_34'], color='#2ca02c', marker='^', markersize=3, label='S33->S34')
ax_p90.plot(weekly_trans['week'], weekly_trans['p90_36_37'], color='#9467bd', marker='d', markersize=3, label='S36->S37')
ax_p90.set_ylabel('90th Percentile Delay (min)', fontsize=11)
ax_p90.set_title('Line 3 Transit Behavior Over Time: 90th Percentile (p90) Inter-Station Delay', fontsize=12, fontweight='bold', loc='left')
ax_p90.legend(loc='upper right', frameon=True)
ax_p90.grid(True, alpha=0.3)

# Panel 2: 99th Percentile Delay (p99) - NO CLIPPING
ax_p99.plot(weekly_trans['week'], weekly_trans['p99_30_33'], color='#d62728', marker='o', markersize=3, label='S30->S33')
ax_p99.plot(weekly_trans['week'], weekly_trans['p99_29_30'], color='#1f77b4', marker='s', markersize=3, label='S29->S30')
ax_p99.plot(weekly_trans['week'], weekly_trans['p99_34_36'], color='#e67e22', marker='v', markersize=3, label='S34->S36')
ax_p99.plot(weekly_trans['week'], weekly_trans['p99_33_34'], color='#2ca02c', marker='^', markersize=3, label='S33->S34')
ax_p99.plot(weekly_trans['week'], weekly_trans['p99_36_37'], color='#9467bd', marker='d', markersize=3, label='S36->S37')
ax_p99.set_ylabel('99th Percentile Delay (min)', fontsize=11)
ax_p99.set_title('99th Percentile (p99) Inter-Station Delay (Unclipped True Full Range)', fontsize=12, fontweight='bold', loc='left')
ax_p99.legend(loc='upper right', frameon=True)
ax_p99.grid(True, alpha=0.3)

# Panel 3: Zero-Delta Share (%)
ax_zero.plot(weekly_trans['week'], weekly_trans['zero_36_37'], color='#9467bd', marker='d', markersize=3, label='S36->S37 (Direct)')
ax_zero.plot(weekly_trans['week'], weekly_trans['zero_33_34'], color='#2ca02c', marker='^', markersize=3, label='S33->S34 (Direct)')
ax_zero.plot(weekly_trans['week'], weekly_trans['zero_29_30'], color='#1f77b4', marker='s', markersize=3, label='S29->S30')
ax_zero.plot(weekly_trans['week'], weekly_trans['zero_34_36'], color='#e67e22', marker='v', markersize=3, label='S34->S36')
ax_zero.plot(weekly_trans['week'], weekly_trans['zero_30_33'], color='#d62728', marker='o', markersize=3, label='S30->S33 (Buffer Queue)')
ax_zero.set_ylabel('Zero-Delta Share (%)', fontsize=11)
ax_zero.set_xlabel('Production Week (1 week = 16.75 relative time units)', fontsize=12)
ax_zero.set_title('Proportion of Parts Transferred in Same Timestamp Bin (0 min Delta)', fontsize=12, fontweight='bold', loc='left')
ax_zero.set_ylim(-2, 102)
ax_zero.legend(loc='lower right', frameon=True)
ax_zero.grid(True, alpha=0.3)

plt.suptitle('Line 3 Inter-Station Pacing Over 103 Weeks: p90, p99, and Zero-Delta Transfer Share', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "task6b_line3_transit_over_time.png"), dpi=300)
plt.close(fig)
print("Saved task6b_line3_transit_over_time.png")
plt.close(fig)
print("Saved task6b_line3_transit_over_time.png")

print("All tasks completed successfully!")
