"""
Real-time Health and Risk Scoring Service for Bosch Line 3 Digital Twin.
Consumes station telemetry events, computes multi-factor station health and cumulative
part failure risks, and publishes health scores, part risks, and line overviews to MQTT.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
import yaml
import paho.mqtt.client as mqtt

from twin_core.schemas import (
    StationTelemetryEvent,
    StationHealthEvent,
    PartRiskEvent,
    LineOverviewEvent,
)
from twin_core.topics import TopicManager
from twin_core.baselines import BaselineManager
from twin_core.windows import StationRollingWindow
from twin_core.health import HealthScorer
from ai.inference import AIInferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScoringService")


class ScoringService:
    """Consumes telemetry, computes evidence-based health/risk, and publishes scored metrics."""

    def __init__(
        self,
        config_path: str = "config/twin_config.yaml",
        scenario: Optional[str] = None,
        enable_ai: Optional[bool] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.scenario = scenario
        self.topic_mgr = TopicManager(self.config["mqtt"]["topics"])
        self.baseline_mgr = BaselineManager(config_path=config_path)
        self.scorer = HealthScorer(self.baseline_mgr)

        # AI Configuration & Services
        self.ai_cfg = self.config.get("ai", {})
        self.ai_enabled = enable_ai if enable_ai is not None else self.ai_cfg.get("enabled", False)
        self.use_ai_risk = self.ai_cfg.get("integration", {}).get("risk", {}).get("use_ai", False)
        self.ai_service: Optional[AIInferenceService] = None

        if self.ai_enabled:
            logger.info(f"Initializing AI Inference Service (Scenario: {self.scenario or 'General'})...")
            self.ai_service = AIInferenceService(models_dir="models", scenario=self.scenario)

        # AI Tracking State
        self.s29_entry_history = deque(maxlen=10000)
        self.current_window_idx: Optional[int] = None
        self.window_dt = 1.0 / 10.02985  # ~0.0997 time units per sim-hour
        self.window_station_metrics: Dict[str, Dict[str, Any]] = {
            s: {"z_scores": [], "transits": [], "zero_deltas": 0, "parts": 0}
            for s in ["S29", "S30", "S33", "S34", "S35", "S36", "S37"]
        }
        self.window_sequence = deque(maxlen=24)

        # Stations in Line 3 core sequence
        self.stations = ["S29", "S30", "S33", "S34", "S35", "S36", "S37"]
        self.station_predecessors = {
            "S29": None,
            "S30": "S29",
            "S33": "S30",
            "S34": "S33",
            "S35": "S34",
            "S36": "S34",
            "S37": None, # Could be S35 or S36 depending on route
        }

        # Rolling windows per station
        window_size = self.config["scoring"]["station_health"]["rolling_window_parts"]
        self.windows: Dict[str, StationRollingWindow] = {
            s: StationRollingWindow(s, max_parts=window_size) for s in self.stations
        }
        # Last known station health scores (station_id -> float)
        self.latest_health: Dict[str, float] = {s: 100.0 for s in self.stations}

        # Part tracking: part_id -> state dict
        # {visited: [stations], anomalies: count, deviations: [...], last_station: str}
        self.active_parts: Dict[int, Dict[str, Any]] = {}

        # Line-level aggregates
        self.total_entered = 0
        self.total_completed = 0
        self.total_defects = 0
        self.recent_completed_times = deque(maxlen=200)

        # MQTT setup
        self.mqtt_cfg = self.config["mqtt"]
        self.mqtt_client = mqtt.Client(
            client_id=f"{self.mqtt_cfg['client_id_prefix']}_scorer",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("Scoring Service connected to MQTT broker. Subscribing to telemetry...")
        client.subscribe("factory/line3/+/telemetry")
        logger.info("Listening for station telemetry events.")

    def _on_message(self, client, userdata, msg):
        try:
            event = StationTelemetryEvent.from_json(msg.payload.decode("utf-8"))
            self.process_telemetry_event(event)
        except Exception as e:
            logger.error(f"Error processing telemetry event: {e}")

    def process_telemetry_event(self, event: StationTelemetryEvent):
        stn = event.station_id
        part_id = event.part_id

        # 1. Update part tracking state
        if part_id not in self.active_parts:
            self.active_parts[part_id] = {
                "visited": [],
                "anomalies": 0,
                "deviations": [],
                "prev_stn": None,
                "features": {},
                "station_times": {},
            }
            if event.is_entry:
                self.total_entered += 1

        part_state = self.active_parts[part_id]
        if stn not in part_state["visited"]:
            part_state["visited"].append(stn)

        # Track features and timestamps
        if event.features:
            part_state["features"].update(event.features)
        part_state["station_times"][stn] = event.sim_time
        if stn == "S29":
            self.s29_entry_history.append(event.sim_time)

        # 1b. Track window aggregation metrics
        if stn in self.window_station_metrics:
            sm = self.window_station_metrics[stn]
            sm["parts"] += 1
            if event.features:
                for fname, val in event.features.items():
                    z = self.baseline_mgr.calculate_z_score(fname, val)
                    sm["z_scores"].append(abs(z))
            if event.transit_minutes is not None:
                sm["transits"].append(event.transit_minutes)
                if event.transit_minutes <= 0.001:
                    sm["zero_deltas"] += 1

        # Check window boundary
        w_idx = int(event.sim_time / self.window_dt)
        if self.current_window_idx is None:
            self.current_window_idx = w_idx
        elif w_idx > self.current_window_idx:
            self._close_window(event.sim_time)
            self.current_window_idx = w_idx

        # Determine predecessor station
        incoming_from = part_state["prev_stn"] or self.station_predecessors.get(stn)
        part_state["prev_stn"] = stn

        # Check feature anomalies for this part
        if event.features:
            for fname, val in event.features.items():
                z = self.baseline_mgr.calculate_z_score(fname, val)
                if abs(z) >= 3.0:
                    part_state["anomalies"] += 1
                    part_state["deviations"].append({
                        "station": stn,
                        "feature": fname,
                        "z_score": round(z, 2),
                    })

        # 2. Evaluate Station Health
        window = self.windows.get(stn, StationRollingWindow(stn))
        health_event = self.scorer.evaluate_station_health(
            station_id=stn,
            incoming_from_station=incoming_from,
            event=event,
            window=window,
        )
        self.latest_health[stn] = health_event.health_score

        # Publish Station Health Event
        health_topic = self.topic_mgr.health_topic(stn)
        self.mqtt_client.publish(health_topic, health_event.to_json(), qos=0)

        # 3. Evaluate Part Risk
        risk_event = self.scorer.evaluate_part_risk(
            part_id=part_id,
            station_id=stn,
            sim_time=event.sim_time,
            visited_stations=part_state["visited"],
            cumulative_anomalies=part_state["anomalies"],
            top_deviations=part_state["deviations"][-5:], # Top recent
            station_health=health_event.health_score,
            ground_truth_response=event.response,
        )

        # AI Defect Risk prediction at S35/S36 decision time
        if self.ai_enabled and self.ai_service is not None and stn in ("S35", "S36"):
            stimes = part_state["station_times"]
            t29 = stimes.get("S29", event.sim_time)
            t30 = stimes.get("S30", t29)
            t33 = stimes.get("S33", t30)
            t34 = stimes.get("S34", t33)

            t_left = event.sim_time - self.window_dt
            s29_tp = sum(1 for t in self.s29_entry_history if t_left <= t <= event.sim_time)
            causal_defect_rate = self.total_defects / max(1, self.total_completed)

            feat_dict = {
                "decision_time": event.sim_time,
                "week": int(event.sim_time / 16.75),
                "branch": 0 if stn == "S35" else 1,
                "transit_29_30": max(0.0, t30 - t29),
                "transit_30_33": max(0.0, t33 - t30),
                "transit_33_34": max(0.0, t34 - t33),
                "transit_34_branch": max(0.0, event.sim_time - t34),
                "s29_throughput_1h": float(s29_tp),
                "recent_defect_rate": float(causal_defect_rate),
            }
            feat_dict.update(part_state["features"])

            pred_prob = self.ai_service.predict_part(feat_dict)
            ai_risk_payload = {
                "part_id": part_id,
                "station_id": stn,
                "branch": 0 if stn == "S35" else 1,
                "decision_time": event.sim_time,
                "predicted_defect_prob": float(pred_prob),
                "is_high_risk": bool(pred_prob >= 0.05),
                "scenario_model": self.scenario or "general",
                "ground_truth_response": event.response,
            }
            self.mqtt_client.publish(
                self.topic_mgr.ai_part_risk_topic(),
                json.dumps(ai_risk_payload),
                qos=0,
            )

            # Incorporate into risk_event if use_ai flag enabled
            if self.use_ai_risk:
                w_cfg = self.ai_cfg.get("integration", {}).get("risk", {}).get("weights", {})
                w_rule = w_cfg.get("rule_risk", 0.30)
                w_stn = w_cfg.get("station_health", 0.20)
                w_pred = w_cfg.get("predicted_p", 0.20)
                rule_risk = risk_event.cumulative_risk
                stn_penalty = (1.0 - health_event.health_score / 100.0)
                p_norm = min(1.0, max(0.0, (pred_prob / 0.00507 - 1.0) / 4.0)) if pred_prob > 0.00507 else 0.0
                composite_risk = (w_rule * rule_risk) + (w_stn * stn_penalty) + (w_pred * p_norm)
                risk_event.cumulative_risk = round(min(1.0, max(0.0, composite_risk)), 3)

        # Publish Part Risk Event
        risk_topic = self.topic_mgr.part_risk_topic(part_id)
        self.mqtt_client.publish(risk_topic, risk_event.to_json(), qos=0)

        # 4. Check for Alerts
        if health_event.status == "CRITICAL":
            alert = {
                "alert_type": "STATION_HEALTH_CRITICAL",
                "station_id": stn,
                "health_score": health_event.health_score,
                "sim_time": event.sim_time,
                "anomalous_features": health_event.anomalous_features,
            }
            self.mqtt_client.publish(self.topic_mgr.alert_topic(), json.dumps(alert), qos=1)

        # 5. Handle Exit at S37
        if event.is_exit:
            self.total_completed += 1
            is_defect = (event.response == 1)
            if is_defect:
                self.total_defects += 1

            # Propagate outcome to the rolling windows of stations this part visited
            for visited_stn in part_state["visited"]:
                if visited_stn in self.windows:
                    self.windows[visited_stn].record_outcome(part_id, 1 if is_defect else 0)

            # Record completion time
            self.recent_completed_times.append(event.sim_time)

            # Calculate rolling throughput (parts per hour)
            throughput_hr = 0.0
            if len(self.recent_completed_times) >= 2:
                dt_sim = self.recent_completed_times[-1] - self.recent_completed_times[0]
                dt_hr = dt_sim * 10.02985
                if dt_hr > 0.01:
                    throughput_hr = round(len(self.recent_completed_times) / dt_hr, 1)

            # Average health across Line 3 stations
            avg_health = round(sum(self.latest_health.values()) / len(self.latest_health), 1)
            cum_defect_rate = round(self.total_defects / max(1, self.total_completed) * 100.0, 3)

            # Publish Line Overview Event
            overview = LineOverviewEvent(
                sim_time=event.sim_time,
                active_parts_in_line=len(self.active_parts),
                total_parts_entered=self.total_entered,
                total_parts_completed=self.total_completed,
                total_defects_observed=self.total_defects,
                cumulative_defect_rate=cum_defect_rate,
                rolling_throughput_per_hr=throughput_hr,
                line_average_health=avg_health,
            )
            self.mqtt_client.publish(self.topic_mgr.overview_topic(), overview.to_json(), qos=0)

            # Clean up memory
            if part_id in self.active_parts:
                del self.active_parts[part_id]

    def _close_window(self, current_sim_time: float):
        """Aggregates metrics for the completed 1-sim-hour window and runs LSTM anomaly model."""
        if not self.ai_enabled or self.ai_service is None:
            return

        window_vector = []
        total_parts_in_window = 0

        for stn in self.stations:
            sm = self.window_station_metrics[stn]
            parts = sm["parts"]
            total_parts_in_window += parts
            zs = sm["z_scores"]
            transits = sm["transits"]
            zero_deltas = sm["zero_deltas"]

            max_z = float(np.max(zs)) if zs else 0.0
            mean_z = float(np.mean(zs)) if zs else 0.0
            p90_t = float(np.percentile(transits, 90)) if transits else 0.0
            zero_share = float(zero_deltas / max(1, len(transits))) if transits else 1.0
            throughput = float(parts)

            # Publish twin/ai/window
            w_payload = {
                "station_id": stn,
                "throughput": throughput,
                "zero_delta_share": zero_share,
                "max_abs_z": max_z,
                "mean_abs_z": mean_z,
                "p90_transit": p90_t,
                "sim_time": current_sim_time,
            }
            self.mqtt_client.publish(
                self.topic_mgr.ai_window_topic(),
                json.dumps(w_payload),
                qos=0,
            )

            # 5 features per station: max_z, mean_z, p90_transit, zero_share, throughput
            window_vector.extend([max_z, mean_z, p90_t, zero_share, throughput])

            # Reset metrics for next window
            self.window_station_metrics[stn] = {
                "z_scores": [], "transits": [], "zero_deltas": 0, "parts": 0
            }

        # If not a gap (< 20 parts) and input dim matches 35
        is_gap = total_parts_in_window < 20
        if not is_gap and len(window_vector) == 35:
            self.window_sequence.append(window_vector)
            if len(self.window_sequence) == 24:
                seq = np.array(self.window_sequence, dtype=np.float32)
                score, contribs, is_flagged = self.ai_service.score_window(seq)
                anom_payload = {
                    "sim_time": current_sim_time,
                    "scenario_model": self.scenario or "general",
                    "anomaly_score": float(score),
                    "threshold_p99": float(self.ai_service.anomaly_threshold),
                    "is_anomaly": bool(is_flagged),
                    "station_contributions": contribs,
                }
                self.mqtt_client.publish(
                    self.topic_mgr.ai_anomaly_topic(),
                    json.dumps(anom_payload),
                    qos=0,
                )

    def start(self):
        try:
            logger.info(f"Connecting to MQTT broker at {self.mqtt_cfg['broker_host']}:{self.mqtt_cfg['broker_port']}...")
            self.mqtt_client.connect(
                self.mqtt_cfg["broker_host"],
                self.mqtt_cfg["broker_port"],
                keepalive=self.mqtt_cfg["keepalive"],
            )
            self.mqtt_client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Scoring service stopped by user.")
        finally:
            self.mqtt_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Run Line 3 Real-Time Health & Risk Scoring Service.")
    parser.add_argument("--config", type=str, default="config/twin_config.yaml", help="Path to config")
    parser.add_argument("--scenario", type=str, default=None, help="Active scenario holdout (A, B, C, D)")
    parser.add_argument("--ai", action="store_true", help="Enable AI models explicitly")
    args = parser.parse_args()

    service = ScoringService(config_path=args.config, scenario=args.scenario, enable_ai=True if args.ai else None)
    service.start()


if __name__ == "__main__":
    main()
