"""
Baseline manager for Bosch Line 3 Digital Twin.
Loads empirical feature distributions, inter-station transit pacing benchmarks,
and line defect rates derived from historical exploration.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import yaml


class BaselineManager:
    """Manages baseline statistics for feature standardization, transit pacing, and defect rates."""

    def __init__(self, config_path: str = "config/twin_config.yaml", features_csv_path: Optional[str] = None):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Baseline parameters
        self.core_defect_rate = self.config["baselines"]["core_flow_defect_rate"]
        self.transit_baselines = self.config["baselines"]["transit_pacing"]
        self.time_config = self.config["time"]
        self.scoring_config = self.config["scoring"]

        # Load informative features stats (mean, std, min, max)
        if features_csv_path is None:
            features_csv_path = os.path.join(
                os.path.dirname(config_path),
                "..",
                "data_prep",
                "output",
                "recommended_subset_features_filtered.csv",
            )
        self.features_csv_path = os.path.abspath(features_csv_path)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.station_features: Dict[str, list] = {
            "S29": [], "S30": [], "S33": [], "S34": [], "S35": [], "S36": [], "S37": []
        }
        self._load_feature_stats()

    def _load_feature_stats(self):
        """Parse recommended_subset_features_filtered.csv."""
        if not os.path.exists(self.features_csv_path):
            raise FileNotFoundError(f"Feature baselines file not found: {self.features_csv_path}")

        df = pd.read_csv(self.features_csv_path)
        for _, row in df.iterrows():
            col = str(row["col_name"])
            stn = str(row["station"])
            stats = {
                "mean": float(row["mean"]),
                "std": float(row["std"]) if float(row["std"]) > 0 else 0.01,
                "min": float(row["min"]),
                "max": float(row["max"]),
                "station": stn,
            }
            self.feature_stats[col] = stats
            if stn in self.station_features:
                self.station_features[stn].append(col)

    def get_feature_stats(self, feature_name: str) -> Optional[Dict[str, float]]:
        return self.feature_stats.get(feature_name)

    def get_station_features(self, station_id: str) -> list:
        return self.station_features.get(station_id, [])

    def calculate_z_score(self, feature_name: str, value: float) -> float:
        """Compute standard Z-score = (x - mean) / std. Returns 0.0 if unknown or invalid."""
        stats = self.feature_stats.get(feature_name)
        if not stats or pd.isna(value):
            return 0.0
        return (value - stats["mean"]) / stats["std"]

    def get_transit_baseline(self, from_stn: str, to_stn: str) -> Dict[str, float]:
        """Lookup transit pacing benchmark for segment (e.g. S29->S30)."""
        key = f"{from_stn}->{to_stn}"
        if key in self.transit_baselines:
            return self.transit_baselines[key]
        # Default fallback
        return {
            "median_min": 0.0,
            "p90_min": 6.0238,
            "p99_min": 36.1060,
            "zero_delta_pct": 50.0,
        }
