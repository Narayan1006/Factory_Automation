"""
CLI runner for Line 3 Replay Engine.
Supports preset scenarios (A, B, C, D) and custom time windows,
with configurable MQTT broker parameters and speed scaling.
"""

import os
import sys
import argparse
import logging
import yaml
import paho.mqtt.client as mqtt

from twin.core.topics import TopicManager
from twin.replay.engine import ReplayEngine
from twin.replay.build_replay_data import build_scenario_data
from twin.paths import TWIN_CONFIG_PATH, REPLAY_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReplayRunner")


def main():
    parser = argparse.ArgumentParser(description="Run Bosch Line 3 Real-Data Replay Engine.")
    parser.add_argument("--scenario", choices=["A", "B", "C", "D"], default="A",
                        help="Pre-configured scenario window (default: A)")
    parser.add_argument("--speed", type=float, default=120.0,
                        help="Simulation speed multiplier (120 = 1 sim-hr in 30s; 0 = burst mode)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit total events to replay (useful for testing)")
    parser.add_argument("--broker", type=str, default="localhost",
                        help="MQTT broker hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=1883,
                        help="MQTT broker port (default: 1883)")
    parser.add_argument("--config", type=str, default=str(TWIN_CONFIG_PATH),
                        help="Path to twin_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without connecting to an MQTT broker (prints progress)")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List available scenarios and exit")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.list_scenarios:
        print("Available Replay Scenarios:")
        for sc_name, sc_data in config.get("scenarios", {}).items():
            print(f"  - Scenario {sc_name}: Window [{sc_data.get('t_start')}, {sc_data.get('t_end')}], Name: {sc_data.get('name')}, Desc: {sc_data.get('description')}")
        return

    topic_mgr = TopicManager(config["mqtt"]["topics"])

    # Ensure dataset exists; build if missing
    dataset_file = os.path.join(str(REPLAY_DATA_DIR), f"scenario_{args.scenario}.parquet")
    if not os.path.exists(dataset_file):
        logger.warning(f"Scenario dataset {dataset_file} not found. Building now from raw data...")
        sc_info = config["scenarios"][args.scenario]
        build_scenario_data(sc_info["t_start"], sc_info["t_end"], f"scenario_{args.scenario}")

    # Setup MQTT Client
    mqtt_client = None
    if not args.dry_run:
        try:
            logger.info(f"Connecting to MQTT broker at {args.broker}:{args.port}...")
            mqtt_client = mqtt.Client(
                client_id=f"{config['mqtt']['client_id_prefix']}_replay",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )
            mqtt_client.connect(args.broker, args.port, keepalive=60)
            mqtt_client.loop_start()
            logger.info("Connected to MQTT broker successfully.")
        except Exception as e:
            logger.warning(
                f"Could not connect to MQTT broker ({e}). "
                "Falling back to dry-run / local mode (no broker required)."
            )
            mqtt_client = None

    engine = ReplayEngine(
        dataset_path=dataset_file,
        topic_manager=topic_mgr,
        speed_multiplier=args.speed,
        mqtt_client=mqtt_client,
    )

    try:
        engine.run(max_events=args.max_events)
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user.")
    finally:
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.info("MQTT connection closed.")


if __name__ == "__main__":
    main()
