"""
Health and risk scoring engine for Line 3 Digital Twin.
Computes deterministic, evidence-based composite scores without ungrounded heuristics.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .schemas import StationTelemetryEvent, StationHealthEvent, PartRiskEvent
from .baselines import BaselineManager
from .windows import StationRollingWindow


class HealthScorer:
    """Calculates composite station health and part cumulative failure risk."""

    def __init__(self, baseline_mgr: BaselineManager):
        self.bm = baseline_mgr
        self.config = baseline_mgr.scoring_config
        self.station_weights = self.config["station_health"]["weights"]
        self.station_thresholds = self.config["station_health"]["thresholds"]
        self.z_thresh = self.config["station_health"]["z_score_threshold"]
        self.part_weights = self.config["part_risk"]
        self.core_defect_rate = self.bm.core_defect_rate

    def evaluate_station_health(
        self,
        station_id: str,
        incoming_from_station: Optional[str],
        event: StationTelemetryEvent,
        window: StationRollingWindow,
    ) -> StationHealthEvent:
        """
        Compute multi-factor composite health score H_s in [0, 100].
        Sub-scores:
          1. feature_drift_score (sensor |Z|-scores)
          2. transit_pacing_score (delay vs p90/p99)
          3. defect_rate_score (observed vs expected failure rate of visiting parts)
        """
        # 1. Feature Drift Score
        anomalous_features = []
        abs_z_list = []
        if event.features:
            for feat_name, val in event.features.items():
                z = self.bm.calculate_z_score(feat_name, val)
                abs_z = abs(z)
                abs_z_list.append(abs_z)
                if abs_z >= self.z_thresh:
                    anomalous_features.append(feat_name)

        if abs_z_list:
            max_abs_z = max(abs_z_list)
            # Rolling max Z
            window.add_visit(event.part_id, event.sim_time, event.transit_minutes, max_abs_z)
            avg_z = window.get_avg_max_z()
            # If avg_z <= 1.0 -> 100 score; if avg_z >= 4.0 -> 0 score
            feature_score = max(0.0, min(100.0, 100.0 - (avg_z - 1.0) * (100.0 / 3.0)))
        else:
            # Stations with 0 informative features (S34, S37) have neutral feature score
            window.add_visit(event.part_id, event.sim_time, event.transit_minutes, 0.0)
            feature_score = 100.0

        # 2. Transit Pacing Score
        if incoming_from_station:
            transit_bench = self.bm.get_transit_baseline(incoming_from_station, station_id)
            p90 = transit_bench["p90_min"]
            p99 = transit_bench["p99_min"]
            t_curr = event.transit_minutes
            if t_curr <= p90:
                pacing_score = 100.0
            elif t_curr <= p99:
                # Interpolate from 100 down to 60
                pacing_score = 100.0 - 40.0 * ((t_curr - p90) / max(0.001, (p99 - p90)))
            else:
                # Greater than p99 -> severe pacing penalty
                pacing_score = max(0.0, 60.0 - 60.0 * ((t_curr - p99) / max(1.0, p99)))
        else:
            # Entry station S29 has no predecessor
            pacing_score = 100.0

        # 3. Defect Rate Score (Observed vs Expected for visiting parts)
        obs_rate = window.get_observed_defect_rate(self.core_defect_rate)
        oe_ratio = obs_rate / max(1e-5, self.core_defect_rate)
        if oe_ratio <= 1.0:
            defect_score = 100.0
        elif oe_ratio <= 3.0:
            # Drop from 100 to 50
            defect_score = 100.0 - 50.0 * ((oe_ratio - 1.0) / 2.0)
        else:
            # Severe defect surge (oe_ratio > 3)
            defect_score = max(0.0, 50.0 - 50.0 * ((oe_ratio - 3.0) / 4.0))

        # Composite weighted sum
        w_f = self.station_weights["feature_drift"]
        w_p = self.station_weights["transit_pacing"]
        w_d = self.station_weights["defect_rate"]

        composite = (w_f * feature_score) + (w_p * pacing_score) + (w_d * defect_score)
        composite = round(max(0.0, min(100.0, composite)), 2)

        # Status categorization
        if composite >= self.station_thresholds["healthy"]:
            status = "HEALTHY"
        elif composite >= self.station_thresholds["warning"]:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return StationHealthEvent(
            station_id=station_id,
            sim_time=event.sim_time,
            health_score=composite,
            feature_drift_score=round(feature_score, 2),
            transit_pacing_score=round(pacing_score, 2),
            defect_rate_score=round(defect_score, 2),
            status=status,
            active_features_count=len(event.features),
            anomalous_features=anomalous_features,
            parts_processed_window=window.get_parts_count(),
            observed_defect_rate=round(obs_rate * 100.0, 4),
        )

    def evaluate_part_risk(
        self,
        part_id: int,
        station_id: str,
        sim_time: float,
        visited_stations: List[str],
        cumulative_anomalies: int,
        top_deviations: List[Dict[str, Any]],
        station_health: float,
        ground_truth_response: Optional[int] = None,
    ) -> PartRiskEvent:
        """
        Evaluate cumulative failure risk for an individual part.
        Risk is bounded in [0.0, 1.0].
        """
        # Feature anomaly component (0.0 to 1.0)
        # 1 anomaly gives ~0.2 risk, 4+ anomalies saturate to ~0.8
        f_risk = min(1.0, cumulative_anomalies * 0.22)

        # Station health penalty (if passing through degraded station)
        h_risk = max(0.0, (100.0 - station_health) / 100.0)

        # Weighted combination
        w_f = self.part_weights["feature_anomaly_weight"]
        w_h = self.part_weights["station_health_penalty_weight"]
        # Normalize weights
        total_w = w_f + w_h
        raw_risk = (w_f * f_risk + w_h * h_risk) / total_w

        # Minimum baseline risk equals the historical defect rate (~0.005)
        risk_score = round(max(self.core_defect_rate, min(0.99, raw_risk)), 4)

        if risk_score < self.part_weights["thresholds"]["low_risk"]:
            risk_level = "LOW"
        elif risk_score < self.part_weights["thresholds"]["high_risk"]:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PartRiskEvent(
            part_id=part_id,
            current_station=station_id,
            sim_time=sim_time,
            cumulative_risk=risk_score,
            risk_level=risk_level,
            stations_visited=list(visited_stations),
            cumulative_anomalies=cumulative_anomalies,
            top_deviations=top_deviations,
            ground_truth_response=ground_truth_response,
        )
