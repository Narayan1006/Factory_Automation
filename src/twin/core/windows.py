"""
Rolling window tracking structures for station health and line throughput calculations.
Maintains bounded deques for real-time statistical aggregations.
"""

from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np


class StationRollingWindow:
    """Rolling window tracker for a single station's recent parts."""

    def __init__(self, station_id: str, max_parts: int = 100):
        self.station_id = station_id
        self.max_parts = max_parts

        # Rolling feature deviations (list of max absolute Z-scores per part)
        self.recent_feature_z_max = deque(maxlen=max_parts)
        # Rolling transit times in minutes
        self.recent_transit_minutes = deque(maxlen=max_parts)
        # Part IDs in recent window
        self.recent_parts = deque(maxlen=max_parts)
        # Timestamps
        self.recent_sim_times = deque(maxlen=max_parts)
        # Known outcomes for parts passing this station (recorded when resolved at S37)
        self.resolved_outcomes: Dict[int, int] = {}
        self.recent_outcomes = deque(maxlen=max_parts)

    def add_visit(self, part_id: int, sim_time: float, transit_minutes: float, max_abs_z: float):
        self.recent_parts.append(part_id)
        self.recent_sim_times.append(sim_time)
        self.recent_transit_minutes.append(transit_minutes)
        self.recent_feature_z_max.append(max_abs_z)

    def record_outcome(self, part_id: int, response: int):
        self.resolved_outcomes[part_id] = response
        if part_id in self.recent_parts:
            self.recent_outcomes.append(response)

    def get_avg_max_z(self) -> float:
        if not self.recent_feature_z_max:
            return 0.0
        return float(np.mean(self.recent_feature_z_max))

    def get_p90_transit(self) -> float:
        if not self.recent_transit_minutes:
            return 0.0
        return float(np.percentile(self.recent_transit_minutes, 90))

    def get_observed_defect_rate(self, baseline_rate: float) -> float:
        if not self.recent_outcomes:
            return baseline_rate
        return float(np.mean(self.recent_outcomes))

    def get_parts_count(self) -> int:
        return len(self.recent_parts)
