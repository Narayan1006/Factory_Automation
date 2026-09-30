import os
import sys
import math
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

BASE_DIR = r"c:\projects\HCL_Projects\PROJECT"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_prep", "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

print("=== 1. FILTERING 64 FEATURES: CHECKING SIGMA AND UNIQUE VALUES ===")
# Load recommended subset features
rec_feats_df = pd.read_csv(os.path.join(OUTPUT_DIR, "recommended_subset_features.csv"))
print(f"Current feature count in recommended_subset_features.csv: {len(rec_feats_df)}")

# Check against sample or compute from train_numeric.csv in chunks
feature_cols = rec_feats_df['col_name'].tolist()

# We already know the 37 retained features from the 64 features.
# Let's inspect S35 numeric features:
s35_cols = ['L3_S35_F3884', 'L3_S35_F3889', 'L3_S35_F3894', 'L3_S35_F3896', 'L3_S31_F3834']
print("Reading sample of S35 and S31 features...")
sample_num = pd.read_csv(os.path.join(DATA_DIR, "train_numeric.csv"), usecols=['Id'] + s35_cols, nrows=50000)
for col in s35_cols:
    sub = sample_num[col].dropna()
    print(f"{col}: count={len(sub)}, std={sub.std():.5f}, unique={sub.nunique()}")


# Let's also check all available numeric features at S35, S34, S36, S37, S31, S33, S29, S30 to see which stations keep informative features
all_meta = pd.read_csv(os.path.join(OUTPUT_DIR, "all_columns_metadata.csv"))
all_meta['line_str'] = all_meta['col_name'].str.extract(r'^(L\d+)_')
all_meta['station_str'] = all_meta['col_name'].str.extract(r'_(S\d+)_')
l3_num = all_meta[(all_meta['line_str'] == 'L3') & (all_meta['file'] == 'numeric')]
print("\nAll L3 numeric feature counts by station:")
print(l3_num.groupby('station_str')['col_name'].count())

print("\n=== 2. S31 ACTIVE TIME SPAN, DEFECT RATE, AND DRIFT INVESTIGATION ===")
part_station_times = pq.read_table(os.path.join(OUTPUT_DIR, "part_station_times.parquet")).to_pandas()
s31_times = part_station_times[part_station_times['station'] == 31]

t_min_s31 = s31_times['entry_time'].min()
t_max_s31 = s31_times['exit_time'].max()
week_min = int(t_min_s31 // 16.75)
week_max = int(t_max_s31 // 16.75)
print(f"S31 parts count: {len(s31_times):,}")
print(f"S31 active time span: t_min = {t_min_s31:.2f} (Week {week_min}), t_max = {t_max_s31:.2f} (Week {week_max})")

# Read responses
part_resp = pq.read_table(os.path.join(OUTPUT_DIR, "part_response.parquet")).to_pandas()
part_times_summary = pq.read_table(os.path.join(OUTPUT_DIR, "part_time_summary.parquet")).to_pandas()
df_parts = part_times_summary.merge(part_resp, on='Id', how='inner')
df_parts['week'] = (df_parts['factory_entry_time'] // 16.75).fillna(-1).astype(int)

# S31 parts
s31_ids = set(s31_times['Id'].unique())
df_parts['visited_s31'] = df_parts['Id'].isin(s31_ids)

# Compare S31 parts vs all parts in the same active weeks
active_weeks = df_parts[df_parts['visited_s31']]['week'].unique()
active_weeks = sorted([w for w in active_weeks if w >= 0])
print(f"S31 active weeks range: min week {min(active_weeks)}, max week {max(active_weeks)}, count of distinct weeks = {len(active_weeks)}")

# Weekly distribution of S31 parts
s31_by_week = df_parts[df_parts['visited_s31']].groupby('week').size()
print("Top 10 weeks with most S31 parts:")
print(s31_by_week.sort_values(ascending=False).head(10))

same_weeks_df = df_parts[df_parts['week'].isin(active_weeks)]
s31_subset = same_weeks_df[same_weeks_df['visited_s31']]
non_s31_subset = same_weeks_df[~same_weeks_df['visited_s31']]

s31_def_rate = s31_subset['Response'].mean() * 100
all_same_weeks_def_rate = same_weeks_df['Response'].mean() * 100
non_s31_def_rate = non_s31_subset['Response'].mean() * 100

print(f"\nDefect rate of S31 parts: {s31_subset['Response'].sum()} / {len(s31_subset)} = {s31_def_rate:.4f}%")
print(f"Defect rate of all parts in same active weeks: {same_weeks_df['Response'].sum()} / {len(same_weeks_df)} = {all_same_weeks_def_rate:.4f}%")
print(f"Defect rate of non-S31 parts in same active weeks: {non_s31_subset['Response'].sum()} / {len(non_s31_subset)} = {non_s31_def_rate:.4f}%")

# Now check L3_S31_F3834 over time: is 7.65 sigma a start-up transient?
print("\nChecking L3_S31_F3834 values over time...")
chunk_size = 100000
s31_meas = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "train_numeric.csv"), usecols=['Id', 'L3_S31_F3834'], chunksize=chunk_size):
    sub = chunk[chunk['L3_S31_F3834'].notna()]
    if len(sub) > 0:
        s31_meas.append(sub)
df_s31_feat = pd.concat(s31_meas, ignore_index=True)
df_s31_feat = df_s31_feat.merge(s31_times[['Id', 'entry_time']], on='Id', how='inner')
df_s31_feat['week'] = (df_s31_feat['entry_time'] // 16.75).astype(int)

mu_s31 = df_s31_feat['L3_S31_F3834'].mean()
std_s31 = df_s31_feat['L3_S31_F3834'].std()
print(f"L3_S31_F3834 overall: mean={mu_s31:.5f}, std={std_s31:.5f}, count={len(df_s31_feat)}")

weekly_s31_z = df_s31_feat.groupby('week')['L3_S31_F3834'].agg(['count', 'mean']).reset_index()
weekly_s31_z['z'] = (weekly_s31_z['mean'] - mu_s31) / std_s31
print("Weekly Z-scores for L3_S31_F3834 (sorted by z descending):")
print(weekly_s31_z.sort_values(by='z', ascending=False).head(10))
print("Weekly Z-scores for L3_S31_F3834 (sorted by week):")
print(weekly_s31_z.sort_values(by='week').head(15))

print("\n=== 3. VERIFY t~366-385 SEQUENCE: S29 DRIFT, DEFECT SURGE, S37 DRIFT ===")
# Inspect t between 360 and 390
# S29 feature L3_S29_F3339, S37 feature L3_S37_F3950, defect rate across Line 3
print("Reading S29 and S37 features in t in [360, 390]...")
s29_times = part_station_times[part_station_times['station'] == 29][['Id', 'entry_time']].rename(columns={'entry_time': 't_s29'})
s37_times = part_station_times[part_station_times['station'] == 37][['Id', 'entry_time']].rename(columns={'entry_time': 't_s37'})

df_seq = df_parts[(df_parts['factory_entry_time'] >= 360) & (df_parts['factory_entry_time'] <= 390)][['Id', 'factory_entry_time', 'factory_exit_time', 'Response']]
print(f"Total parts entering factory in t in [360, 390]: {len(df_seq)}")
print(f"Defects in this period: {df_seq['Response'].sum()} ({df_seq['Response'].mean()*100:.3f}%)")

# Let's read F3339 and F3950 for these parts
sample_feats = []
for chunk in pd.read_csv(os.path.join(DATA_DIR, "train_numeric.csv"), usecols=['Id', 'L3_S29_F3339', 'L3_S37_F3950'], chunksize=chunk_size):
    sub = chunk[chunk['Id'].isin(df_seq['Id'])]
    if len(sub) > 0:
        sample_feats.append(sub)
df_seq_feats = pd.concat(sample_feats, ignore_index=True)
df_seq = df_seq.merge(df_seq_feats, on='Id', how='left').merge(s29_times, on='Id', how='left').merge(s37_times, on='Id', how='left')

print(f"Sequence dataframe merged: {len(df_seq)} rows")
print(df_seq[['factory_entry_time', 't_s29', 't_s37', 'L3_S29_F3339', 'L3_S37_F3950', 'Response']].describe())


