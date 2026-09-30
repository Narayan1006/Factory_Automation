"""
Deterministic real-data replay engine for Line 3 Digital Twin.
Replays chronological station visits at configurable simulation speed multipliers,
serializes StationTelemetryEvents, and publishes to MQTT topics.
"""

import time
import json
import logging
from typing import Optional, Callable, Dict, Any
import pandas as pd
import paho.mqtt.client as mqtt

from twin.core.schemas import StationTelemetryEvent
from twin.core.topics import TopicManager
from twin.core.baselines import BaselineManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReplayEngine")

SECONDS_PER_UNIT = 36107.46268656716  # 10.02985 hours * 3600 seconds


class ReplayEngine:
    """Streams historical factory telemetry events according to their empirical timeline."""

    def __init__(
        self,
        dataset_path: str,
        topic_manager: TopicManager,
        speed_multiplier: float = 120.0,
        mqtt_client: Optional[mqtt.Client] = None,
        event_callback: Optional[Callable[[StationTelemetryEvent], None]] = None,
    ):
        self.dataset_path = dataset_path
        self.topic_mgr = topic_manager
        self.speed = float(speed_multiplier)
        self.client = mqtt_client
        self.callback = event_callback
        self.is_running = False

        # Load dataset
        logger.info(f"Loading replay events from {dataset_path}...")
        self.df_events = pd.read_parquet(dataset_path)
        logger.info(f"Loaded {len(self.df_events):,} events ready for simulation.")

    def run(self, max_events: Optional[int] = None):
        """Execute the replay loop."""
        self.is_running = True
        total = len(self.df_events)
        if max_events is not None:
            total = min(total, max_events)

        logger.info(
            f"Starting replay: {total:,} events at speed {self.speed}x "
            f"({'BURST MODE (no delay)' if self.speed <= 0 else f'1 sim-hr = {3600/self.speed:.2f}s wall-time'})"
        )

        prev_sim_time = None
        count = 0
        t_start_wall = time.time()

        for idx, row in self.df_events.head(total).iterrows():
            if not self.is_running:
                logger.info("Replay stopped by user or signal.")
                break

            curr_sim_time = float(row["sim_time"])

            # Pace replay according to delta sim_time if speed > 0
            if prev_sim_time is not None and self.speed > 0:
                delta_sim = curr_sim_time - prev_sim_time
                if delta_sim > 0:
                    wall_sleep = (delta_sim * SECONDS_PER_UNIT) / self.speed
                    # Cap maximum sleep to 5 seconds to avoid freezing on massive production gaps during demo
                    if wall_sleep > 5.0:
                        logger.info(
                            f"Production gap detected: Delta sim_time={delta_sim:.4f} units "
                            f"(~{delta_sim * 10.03:.1f}h). Fast-forwarding gap in 5.0s."
                        )
                        wall_sleep = 5.0
                    if wall_sleep > 0.001:
                        time.sleep(wall_sleep)

            prev_sim_time = curr_sim_time

            # Parse features
            raw_features = json.loads(row["features_json"]) if isinstance(row["features_json"], str) else {}
            features = {k: float(v) for k, v in raw_features.items()}

            # Quality response: revealed ONLY at S37 exit
            is_exit = bool(row["is_exit"])
            response_val = int(row["response"]) if is_exit and row["response"] >= 0 else None

            # Build standard telemetry event
            event = StationTelemetryEvent(
                part_id=int(row["part_id"]),
                station_id=str(row["station_id"]),
                line_id=3,
                sim_time=curr_sim_time,
                sim_time_hours=float(row["sim_time_hours"]),
                transit_minutes=float(row["transit_minutes"]),
                features=features,
                is_entry=bool(row["is_entry"]),
                is_exit=is_exit,
                response=response_val,
                metadata={"ground_truth": int(row["ground_truth_label"])},
            )

            # Publish to MQTT
            topic = self.topic_mgr.telemetry_topic(event.station_id)
            payload = event.to_json()

            if self.client is not None and self.client.is_connected():
                self.client.publish(topic, payload, qos=0)

            # Invoke local callback if registered
            if self.callback is not None:
                self.callback(event)

            count += 1
            if count % 1000 == 0 or count == total:
                elapsed_wall = time.time() - t_start_wall
                eps = count / max(0.001, elapsed_wall)
                logger.info(
                    f"Progress: {count:,}/{total:,} events ({count/total*100:.1f}%) | "
                    f"Current sim_time: {curr_sim_time:.2f} units | Replay rate: {eps:.1f} events/s"
                )

        total_elapsed = time.time() - t_start_wall
        logger.info(f"Replay completed: {count:,} events in {total_elapsed:.2f} seconds.")

    def stop(self):
        self.is_running = False
