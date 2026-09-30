"""
Data schemas and serialization for Line 3 Digital Twin events.
Used across Replay, MQTT transmission, InfluxDB ingestion, and Scoring.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json


@dataclass
class StationTelemetryEvent:
    """Event emitted when a part arrives at or is processed by a station."""
    part_id: int
    station_id: str
    line_id: int
    sim_time: float                # Simulation time in relative units (e.g. 362.45)
    sim_time_hours: float          # Converted to estimated hours (sim_time * 10.02985)
    transit_minutes: float         # Time elapsed from previous station in minutes (0.0 for S29 entry)
    features: Dict[str, float]     # Informative numeric features present for this station
    is_entry: bool = False         # True if S29
    is_exit: bool = False          # True if S37
    response: Optional[int] = None # Final QC label (only revealed at S37 exit, None otherwise)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StationTelemetryEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "StationTelemetryEvent":
        return cls.from_dict(json.loads(payload))


@dataclass
class StationHealthEvent:
    """Real-time health assessment published for an individual station."""
    station_id: str
    sim_time: float
    health_score: float            # 0 to 100 (100 = optimal, <80 warning, <50 critical)
    feature_drift_score: float     # 0 to 100 component (sensor deviations)
    transit_pacing_score: float    # 0 to 100 component (delay pacing)
    defect_rate_score: float       # 0 to 100 component (rolling failure rate of visiting parts)
    status: str                    # 'HEALTHY', 'WARNING', 'CRITICAL'
    active_features_count: int
    anomalous_features: List[str]  # List of features currently breaching |Z| > 3
    parts_processed_window: int
    observed_defect_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StationHealthEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "StationHealthEvent":
        return cls.from_dict(json.loads(payload))


@dataclass
class PartRiskEvent:
    """Cumulative risk assessment published for a part moving through Line 3."""
    part_id: int
    current_station: str
    sim_time: float
    cumulative_risk: float         # 0.0 to 1.0 (estimated probability/risk of failure)
    risk_level: str                # 'LOW', 'MEDIUM', 'HIGH'
    stations_visited: List[str]
    cumulative_anomalies: int
    top_deviations: List[Dict[str, Any]]
    ground_truth_response: Optional[int] = None # Available at S37 exit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartRiskEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "PartRiskEvent":
        return cls.from_dict(json.loads(payload))


@dataclass
class LineOverviewEvent:
    """Aggregate snapshot of overall Line 3 performance."""
    sim_time: float
    active_parts_in_line: int
    total_parts_entered: int
    total_parts_completed: int
    total_defects_observed: int
    cumulative_defect_rate: float
    rolling_throughput_per_hr: float
    line_average_health: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineOverviewEvent":
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "LineOverviewEvent":
        return cls.from_dict(json.loads(payload))
