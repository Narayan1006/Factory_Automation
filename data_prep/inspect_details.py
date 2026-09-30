"""
Inspect specifics requested by the user:
1. Exact counts for S29->S30->S33->S34->S36->S37 and ...->S35->S37
2. Role and route placement of S31
3. 582 unrouted parts
4. S29 arrival rate, batch sizes, inter-arrivals
5. S24/S25 duration percentiles (p50, p90, p99) and extreme parts
6. Transit times between consecutive Line 3 stations
"""
import pandas as pd
import numpy as np

# Load tables
routes_df = pd.read_parquet('data_prep/output/part_routes.parquet')
events_df = pd.read_parquet('data_prep/output/part_station_times.parquet')
resp_df = pd.read_parquet('data_prep/output/part_response.parquet')

total_parts = 1183747
total_routed = len(routes_df)
print(f"Total parts: {total_parts:,}, total routed: {total_routed:,}, unrouted: {total_parts - total_routed}")

# Unrouted parts check
unrouted_ids = set(resp_df['Id']) - set(routes_df['Id'])
unrouted_resp = resp_df[resp_df['Id'].isin(unrouted_ids)]
print(f"582 unrouted parts: failure count = {unrouted_resp['Response'].sum()}, failure rate = {unrouted_resp['Response'].mean()*100:.4f}%")

# Line 3 exact sequences
routes_df['has_s36_seq'] = routes_df['route'].str.contains('S29->S30->S33->S34->S36->S37', regex=False)
routes_df['has_s35_seq'] = routes_df['route'].str.contains('S29->S30->S33->S34->S35->S37', regex=False)
routes_df['has_s31'] = routes_df['route'].str.contains('S31', regex=False)
routes_df['has_s31_seq'] = routes_df['route'].str.contains('S29->S30->S31->S33', regex=False)

n_s36 = routes_df['has_s36_seq'].sum()
n_s35 = routes_df['has_s35_seq'].sum()
n_both_seq = (routes_df['has_s36_seq'] | routes_df['has_s35_seq']).sum()

def_s36 = routes_df.loc[routes_df['has_s36_seq'], 'Response'].sum()
def_s35 = routes_df.loc[routes_df['has_s35_seq'], 'Response'].sum()
def_both = routes_df.loc[routes_df['has_s36_seq'] | routes_df['has_s35_seq'], 'Response'].sum()

print("\n--- Route Coverage Exact Breakdown ---")
print(f"S29->S30->S33->S34->S36->S37: {n_s36:,} parts ({n_s36/total_parts*100:.2f}%) | Defects: {def_s36:,} ({def_s36/n_s36*100:.4f}%)")
print(f"S29->S30->S33->S34->S35->S37: {n_s35:,} parts ({n_s35/total_parts*100:.2f}%) | Defects: {def_s35:,} ({def_s35/n_s35*100:.4f}%)")
print(f"Combined S29->S30->S33->S34->(S35 or S36)->S37: {n_both_seq:,} parts ({n_both_seq/total_parts*100:.2f}%) | Defects: {def_both:,} ({def_both/n_both_seq*100:.4f}%)")

# S31 details
n_s31 = routes_df['has_s31'].sum()
n_s31_seq = routes_df['has_s31_seq'].sum()
def_s31 = routes_df.loc[routes_df['has_s31'], 'Response'].sum()
print(f"\nS31 presence: {n_s31:,} parts ({n_s31/total_parts*100:.2f}%) | S29->S30->S31->S33: {n_s31_seq:,} parts ({n_s31_seq/n_s31*100:.1f}% of S31 parts) | Defects: {def_s31:,} ({def_s31/n_s31*100:.4f}%)")

# S29 arrival dynamics
s29_events = events_df[events_df['station'] == 29].sort_values('entry_time')
s29_times = s29_events['entry_time'].values
print(f"\n--- S29 Specifically ---")
print(f"Total parts at S29: {len(s29_times):,}")
t_min, t_max = s29_times.min(), s29_times.max()
HOURS_PER_UNIT = 168.0 / 16.75
MINUTES_PER_UNIT = HOURS_PER_UNIT * 60.0
s29_span_hours = (t_max - t_min) * HOURS_PER_UNIT
s29_rate_per_hour = len(s29_times) / s29_span_hours
print(f"Time span at S29: {s29_span_hours:.1f} hours ({s29_span_hours/168.0:.1f} weeks)")
print(f"S29 arrival rate: {s29_rate_per_hour:.2f} parts/hour ({s29_rate_per_hour*24:.1f} parts/day)")

# Batch size at S29 (number of parts arriving at identical timestamp)
time_counts = pd.Series(s29_times).value_counts()
print(f"S29 distinct timestamp bins: {len(time_counts):,}")
print(f"S29 batch size (parts sharing identical timestamp): mean = {time_counts.mean():.2f}, median = {time_counts.median():.0f}, p90 = {time_counts.quantile(0.90):.0f}, max = {time_counts.max()}")

# S24 and S25 robust duration baselines
print("\n--- S24 & S25 Robust Duration Baselines ---")
for st in [24, 25]:
    st_ev = events_df[events_df['station'] == st]
    dur_min = st_ev['duration'].values * MINUTES_PER_UNIT
    p50 = np.percentile(dur_min, 50)
    p90 = np.percentile(dur_min, 90)
    p95 = np.percentile(dur_min, 95)
    p99 = np.percentile(dur_min, 99)
    print(f"Station S{st} Duration (minutes): median(p50)={p50:.2f}, p90={p90:.2f}, p95={p95:.2f}, p99={p99:.2f}, mean={np.mean(dur_min):.2f}, max={np.max(dur_min):.2f}")

# Line 3 inter-station transit/wait times
print("\n--- Line 3 Inter-Station Transit/Wait Time (Entry delta between consecutive stations) ---")
# Pivot or merge for parts visiting consecutive stations
p_times = events_df[events_df['station'].isin([29, 30, 31, 33, 34, 35, 36, 37])].pivot(index='Id', columns='station', values='entry_time')

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

for label, s_from, s_to in transit_pairs:
    if s_from in p_times.columns and s_to in p_times.columns:
        deltas = (p_times[s_to] - p_times[s_from]).dropna()
        deltas_min = deltas * MINUTES_PER_UNIT
        # Positive only
        pos_deltas = deltas_min[deltas_min >= 0]
        zero_pct = (deltas_min == 0).mean() * 100
        print(f"{label:<12}: count={len(pos_deltas):>8,}, median={np.median(pos_deltas):>6.2f} min, p90={np.percentile(pos_deltas, 90):>6.2f} min, p99={np.percentile(pos_deltas, 99):>7.2f} min, 0-delta={zero_pct:>5.1f}%")
