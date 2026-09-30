"""
MQTT to InfluxDB Ingestion Service for Bosch Line 3 Digital Twin.
Subscribes to telemetry, health, risk, and line overview topics,
converts events into InfluxDB Line Protocol points, and writes them to InfluxDB v2.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, Any, Optional
import yaml
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from twin.paths import TWIN_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MQTTToInflux")


class IngestionService:
    """Consumes twin events from MQTT and commits time-series data to InfluxDB v2."""

    def __init__(self, config_path: Optional[str] = None, dry_run: bool = False):
        cfg_file = str(config_path) if config_path is not None else str(TWIN_CONFIG_PATH)
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.dry_run = dry_run
        self.mqtt_cfg = self.config["mqtt"]
        self.influx_cfg = self.config["influxdb"]

        # Base reference timestamp (sim_time 0.0 maps to this epoch)
        # Using a fixed epoch base so queries align cleanly
        self.base_epoch_ns = int(time.time() * 1e9)
        self.seconds_per_unit = 36107.46268656716  # 10.02985 h * 3600 s

        # InfluxDB Client
        self.influx_client: Optional[InfluxDBClient] = None
        self.write_api = None
        if not self.dry_run:
            self._connect_influx()

        # MQTT Client
        self.mqtt_client = mqtt.Client(
            client_id=f"{self.mqtt_cfg['client_id_prefix']}_ingest",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message

        self.points_written = 0

    def _connect_influx(self):
        try:
            logger.info(f"Connecting to InfluxDB at {self.influx_cfg['url']}...")
            self.influx_client = InfluxDBClient(
                url=self.influx_cfg["url"],
                token=self.influx_cfg["token"],
                org=self.influx_cfg["org"],
            )
            # Ping test
            ready = self.influx_client.ping()
            if ready:
                logger.info("Connected to InfluxDB successfully.")
                self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            else:
                logger.warning("InfluxDB ping failed. Operating in fallback log mode.")
        except Exception as e:
            logger.warning(f"Could not connect to InfluxDB ({e}). Operating in log-only mode.")
            self.influx_client = None
            self.write_api = None

    def _sim_time_to_timestamp_ns(self, sim_time: float) -> int:
        """Convert relative simulation time to nano epoch for time-series alignment."""
        sim_seconds = sim_time * self.seconds_per_unit
        return self.base_epoch_ns + int(sim_seconds * 1e9)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("Connected to MQTT broker. Subscribing to factory topics...")
        # Subscribe to all Line 3 topics
        client.subscribe("factory/line3/+/telemetry")
        client.subscribe("factory/line3/+/health")
        client.subscribe("factory/line3/parts/+/risk")
        client.subscribe("factory/line3/overview")
        client.subscribe("factory/line3/alerts")
        client.subscribe("twin/ai/#")
        logger.info("Subscriptions active.")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8")
        try:
            data = json.loads(payload_str)
            point = self._convert_to_point(topic, data)
            if point is not None:
                if self.write_api is not None:
                    self.write_api.write(
                        bucket=self.influx_cfg["bucket"],
                        org=self.influx_cfg["org"],
                        record=point,
                    )
                self.points_written += 1
                if self.points_written % 1000 == 0:
                    logger.info(f"Ingested {self.points_written:,} time-series points into InfluxDB.")
        except Exception as e:
            logger.error(f"Error ingesting message from topic {topic}: {e}")

    def _convert_to_point(self, topic: str, data: Dict[str, Any]) -> Optional[Point]:
        sim_time = float(data.get("sim_time", 0.0))
        # Use current wall-clock nano timestamp for live Grafana real-time display
        t_ns = time.time_ns()

        # Handle AI measurements first (twin/ai/...)
        if "twin/ai/part_risk" in topic:
            p = Point("ai_part_risk") \
                .tag("part_id", str(data.get("part_id", 0))) \
                .tag("station_id", str(data.get("station_id", "S35"))) \
                .tag("branch", str(data.get("branch", "0"))) \
                .tag("scenario_model", str(data.get("scenario_model", "general"))) \
                .tag("is_high_risk", str(data.get("is_high_risk", False))) \
                .field("predicted_defect_prob", float(data.get("predicted_defect_prob", 0.0))) \
                .field("decision_time", float(data.get("decision_time", 0.0)))

            if data.get("raw_defect_prob") is not None:
                p = p.field("raw_defect_prob", float(data["raw_defect_prob"]))
            if data.get("ground_truth_response") is not None:
                p = p.field("ground_truth_response", int(data["ground_truth_response"]))
            p.time(t_ns, WritePrecision.NS)
            return p

        elif "twin/ai/window" in topic:
            p = Point("ai_window") \
                .tag("station_id", str(data.get("station_id", "S29"))) \
                .field("throughput", float(data.get("throughput", 0.0))) \
                .field("zero_delta_share", float(data.get("zero_delta_share", 0.0))) \
                .field("max_abs_z", float(data.get("max_abs_z", 0.0))) \
                .field("mean_abs_z", float(data.get("mean_abs_z", 0.0))) \
                .field("p90_transit", float(data.get("p90_transit", 0.0))) \
                .time(t_ns, WritePrecision.NS)
            return p

        elif "twin/ai/anomaly" in topic:
            p = Point("ai_anomaly") \
                .tag("scenario_model", str(data.get("scenario_model", "general"))) \
                .tag("is_anomaly", str(data.get("is_anomaly", False))) \
                .field("anomaly_score", float(data.get("anomaly_score", 0.0))) \
                .field("threshold_p99", float(data.get("threshold_p99", 0.0)))

            for stn, attr_val in data.get("station_contributions", {}).items():
                p = p.field(f"attr_{stn}", float(attr_val))
            p.time(t_ns, WritePrecision.NS)
            return p

        if "telemetry" in topic:
            p = Point("station_telemetry") \
                .tag("station_id", str(data.get("station_id"))) \
                .tag("line_id", "3") \
                .tag("is_entry", str(data.get("is_entry", False))) \
                .tag("is_exit", str(data.get("is_exit", False))) \
                .field("part_id", int(data.get("part_id", 0))) \
                .field("sim_time", sim_time) \
                .field("sim_time_hours", float(data.get("sim_time_hours", 0.0))) \
                .field("transit_minutes", float(data.get("transit_minutes", 0.0)))

            if data.get("response") is not None:
                p = p.field("response", int(data["response"]))

            # Add sensor feature measurements
            features = data.get("features", {})
            for fname, val in features.items():
                if val is not None:
                    p = p.field(fname, float(val))
            p.time(t_ns, WritePrecision.NS)
            return p

        elif "health" in topic:
            p = Point("station_health") \
                .tag("station_id", str(data.get("station_id"))) \
                .tag("status", str(data.get("status", "HEALTHY"))) \
                .field("health_score", float(data.get("health_score", 100.0))) \
                .field("feature_drift_score", float(data.get("feature_drift_score", 100.0))) \
                .field("transit_pacing_score", float(data.get("transit_pacing_score", 100.0))) \
                .field("defect_rate_score", float(data.get("defect_rate_score", 100.0))) \
                .field("active_features_count", int(data.get("active_features_count", 0))) \
                .field("parts_processed_window", int(data.get("parts_processed_window", 0))) \
                .field("observed_defect_rate", float(data.get("observed_defect_rate", 0.0))) \
                .time(t_ns, WritePrecision.NS)
            return p

        elif "risk" in topic:
            p = Point("part_risk") \
                .tag("part_id", str(data.get("part_id"))) \
                .tag("current_station", str(data.get("current_station"))) \
                .tag("risk_level", str(data.get("risk_level", "LOW"))) \
                .field("cumulative_risk", float(data.get("cumulative_risk", 0.0))) \
                .field("cumulative_anomalies", int(data.get("cumulative_anomalies", 0)))

            if data.get("ground_truth_response") is not None:
                p = p.field("ground_truth_response", int(data["ground_truth_response"]))
            p.time(t_ns, WritePrecision.NS)
            return p

        elif "overview" in topic:
            p = Point("line_overview") \
                .tag("line_id", "3") \
                .field("active_parts_in_line", int(data.get("active_parts_in_line", 0))) \
                .field("total_parts_entered", int(data.get("total_parts_entered", 0))) \
                .field("total_parts_completed", int(data.get("total_parts_completed", 0))) \
                .field("total_defects_observed", int(data.get("total_defects_observed", 0))) \
                .field("cumulative_defect_rate", float(data.get("cumulative_defect_rate", 0.0))) \
                .field("rolling_throughput_per_hr", float(data.get("rolling_throughput_per_hr", 0.0))) \
                .field("line_average_health", float(data.get("line_average_health", 100.0))) \
                .time(t_ns, WritePrecision.NS)
            return p

        return None

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
            logger.info("Ingestion service stopped by user.")
        finally:
            if self.write_api is not None:
                self.write_api.close()
            if self.influx_client is not None:
                self.influx_client.close()
            self.mqtt_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Run Line 3 MQTT to InfluxDB Ingestion Service.")
    parser.add_argument("--config", type=str, default=str(TWIN_CONFIG_PATH), help="Path to config")
    parser.add_argument("--dry-run", action="store_true", help="Log without writing to InfluxDB")
    args = parser.parse_args()

    service = IngestionService(config_path=args.config, dry_run=args.dry_run)
    service.start()


if __name__ == "__main__":
    main()
