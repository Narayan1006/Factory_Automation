"""
End-to-End Integration Test for Bosch Line 3 Digital Twin (Phase 1).
Validates configuration loading, replay datasets (A, B, C, D),
scoring service mechanics, and InfluxDB point conversions without external dependencies.
"""

import os
import sys
import unittest
import json
import pandas as pd

from twin_core import (
    TopicManager,
    BaselineManager,
    HealthScorer,
    StationTelemetryEvent,
    StationRollingWindow,
)
from replay.engine import ReplayEngine
from scoring.scoring_service import ScoringService
from ingest.mqtt_to_influx import IngestionService


class TestBoschDigitalTwinPhase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = "config/twin_config.yaml"
        cls.bm = BaselineManager(cls.config_path)
        cls.scorer = HealthScorer(cls.bm)

    def test_01_configuration_and_baselines(self):
        """Verify that all baseline files and informative features load correctly."""
        self.assertEqual(len(self.bm.feature_stats), 39, "Must load exactly 39 filtered features")
        self.assertEqual(len(self.bm.get_station_features("S29")), 17)
        self.assertEqual(len(self.bm.get_station_features("S30")), 10)
        self.assertEqual(len(self.bm.get_station_features("S33")), 6)
        self.assertEqual(len(self.bm.get_station_features("S35")), 2)
        self.assertEqual(len(self.bm.get_station_features("S36")), 4)
        self.assertEqual(len(self.bm.get_station_features("S34")), 0)
        self.assertEqual(len(self.bm.get_station_features("S37")), 0)

        # Check transit baselines
        pacing_29_30 = self.bm.get_transit_baseline("S29", "S30")
        self.assertIn("p90_min", pacing_29_30)
        self.assertAlmostEqual(pacing_29_30["median_min"], 0.0, places=2)

    def test_02_scenario_datasets_exist(self):
        """Verify all 4 scenario datasets exist with positive event counts."""
        scenarios = ["A", "B", "C", "D"]
        for sc in scenarios:
            path = os.path.join("replay_data", f"scenario_{sc}.parquet")
            self.assertTrue(os.path.exists(path), f"Scenario dataset {path} must exist")
            df = pd.read_parquet(path)
            self.assertGreater(len(df), 1000, f"Scenario {sc} must contain events")
            # Verify columns
            required_cols = [
                "part_id", "station_id", "sim_time", "transit_minutes",
                "features_json", "is_entry", "is_exit", "response", "ground_truth_label"
            ]
            for col in required_cols:
                self.assertIn(col, df.columns)

    def test_03_scoring_service_logic(self):
        """Test that scoring service correctly evaluates station health and part risk."""
        scoring = ScoringService(self.config_path)

        # Create simulated events for a part traversing S29 -> S30 -> S33 -> S34 -> S36 -> S37
        events = [
            StationTelemetryEvent(
                part_id=9999, station_id="S29", line_id=3, sim_time=365.0,
                sim_time_hours=3660.9, transit_minutes=0.0, features={"L3_S29_F3315": 0.55}, # Anomaly
                is_entry=True, is_exit=False, response=None
            ),
            StationTelemetryEvent(
                part_id=9999, station_id="S30", line_id=3, sim_time=365.0,
                sim_time_hours=3660.9, transit_minutes=0.0, features={},
                is_entry=False, is_exit=False, response=None
            ),
            StationTelemetryEvent(
                part_id=9999, station_id="S37", line_id=3, sim_time=365.05,
                sim_time_hours=3661.4, transit_minutes=30.0, features={},
                is_entry=False, is_exit=True, response=1 # Exit with defect
            ),
        ]

        for ev in events:
            scoring.process_telemetry_event(ev)

        # Verify part tracking updated
        self.assertEqual(scoring.total_entered, 1)
        self.assertEqual(scoring.total_completed, 1)
        self.assertEqual(scoring.total_defects, 1)

        # Verify S29 health dropped due to feature anomaly
        s29_health = scoring.latest_health["S29"]
        self.assertLess(s29_health, 80.0, "S29 health should drop below 80 upon severe feature anomaly")

    def test_04_influx_point_generation(self):
        """Test that IngestionService correctly serializes events into InfluxDB Points."""
        ingest = IngestionService(self.config_path, dry_run=True)

        telemetry_payload = {
            "part_id": 1234,
            "station_id": "S29",
            "sim_time": 365.12,
            "sim_time_hours": 3662.1,
            "transit_minutes": 0.0,
            "features": {"L3_S29_F3315": 0.12},
            "is_entry": True,
            "is_exit": False,
        }
        point = ingest._convert_to_point("factory/line3/S29/telemetry", telemetry_payload)
        self.assertIsNotNone(point)
        line_protocol = point.to_line_protocol()
        self.assertIn("station_telemetry", line_protocol)
        self.assertIn("station_id=S29", line_protocol)
        self.assertIn("L3_S29_F3315=0.12", line_protocol)

        health_payload = {
            "station_id": "S29",
            "sim_time": 365.12,
            "health_score": 75.5,
            "feature_drift_score": 60.0,
            "transit_pacing_score": 100.0,
            "defect_rate_score": 90.0,
            "status": "WARNING",
            "active_features_count": 1,
            "parts_processed_window": 50,
            "observed_defect_rate": 0.5069,
        }
        h_point = ingest._convert_to_point("factory/line3/S29/health", health_payload)
        self.assertIsNotNone(h_point)
        h_line = h_point.to_line_protocol()
        self.assertIn("station_health", h_line)
        self.assertIn("status=WARNING", h_line)
        self.assertIn("health_score=75.5", h_line)


if __name__ == "__main__":
    unittest.main()
