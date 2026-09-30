"""
MQTT topic conventions for Bosch Line 3 Digital Twin.
Strictly hierarchical, predictable, and aligned with standard ISA-95 / IIoT patterns.
"""

from typing import Dict, Any


class TopicManager:
    """Manages topic formatting and parsing for MQTT broker communication."""

    def __init__(self, topic_config: Dict[str, str]):
        self.station_telemetry_template = topic_config.get(
            "station_telemetry", "factory/line3/{station_id}/telemetry"
        )
        self.station_health_template = topic_config.get(
            "station_health", "factory/line3/{station_id}/health"
        )
        self.part_risk_template = topic_config.get(
            "part_risk", "factory/line3/parts/{part_id}/risk"
        )
        self.line_overview_topic = topic_config.get(
            "line_overview", "factory/line3/overview"
        )
        self.alerts_topic = topic_config.get(
            "alerts", "factory/line3/alerts"
        )

    def telemetry_topic(self, station_id: str) -> str:
        """Topic for individual station sensor readings and transit events."""
        return self.station_telemetry_template.format(station_id=station_id)

    def health_topic(self, station_id: str) -> str:
        """Topic for station composite health scores and sub-metrics."""
        return self.station_health_template.format(station_id=station_id)

    def part_risk_topic(self, part_id: int) -> str:
        """Topic for individual part cumulative quality risk assessment."""
        return self.part_risk_template.format(part_id=part_id)

    def overview_topic(self) -> str:
        """Topic for overall line throughput, WIP, and active status."""
        return self.line_overview_topic

    def alert_topic(self) -> str:
        """Topic for critical alerts (e.g. Health < 50, stoppage detected)."""
        return self.alerts_topic

    def ai_part_risk_topic(self) -> str:
        """Topic for part-level AI defect risk predictions."""
        return "twin/ai/part_risk"

    def ai_window_topic(self) -> str:
        """Topic for station window telemetry aggregations."""
        return "twin/ai/window"

    def ai_anomaly_topic(self) -> str:
        """Topic for LSTM multivariate anomaly detection and attribution."""
        return "twin/ai/anomaly"

    @staticmethod
    def parse_station_from_topic(topic: str) -> str:
        """Extract station_id from topic like factory/line3/S29/telemetry."""
        parts = topic.split("/")
        if len(parts) >= 4 and parts[0] == "factory" and parts[1] == "line3":
            return parts[2]
        return ""
