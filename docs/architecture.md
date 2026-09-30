# System Architecture: Bosch Line 3 Digital Twin

## 1. High-Level Architecture Overview

The system is organized into four loosely-coupled layers communicating via industrial Pub/Sub messaging (MQTT) and persistent time-series telemetry (InfluxDB v2).

```
                      +-----------------------------------------------+
                      |          HISTORICAL REPLAY ENGINE             |
                      |  Reads pre-compiled Parquet scenario data     |
                      |  Simulates discrete-event clock (1x - 120x)   |
                      +-----------------------------------------------+
                                             │
                                             ▼ (MQTT: factory/line3/{stn}/telemetry)
      +─────────────────────────────────────────────────────────────────────────────+
      │                                                                             │
      ▼                                                                             ▼
+──────────────────────────────+                              +──────────────────────────────+
|   SCORING & AI SERVICE       |                              |   INGESTION SERVICE          |
| - Rule-based Health Scorer   |                              | - Subscribes to all topics   |
| - Part Risk Estimator        |                              | - Converts JSON to Points    |
| - PyTorch Defect MLP (GPU)   |                              | - Batch writes Line Protocol |
| - LSTM Autoencoder (GPU)     |                              +──────────────────────────────+
+──────────────────────────────+                                             │
      │                                                                      │
      ▼ (MQTT: factory/line3/{health, risk, ai/*})                           ▼
+────────────────────────────────────────────────────────────────────────────────────────────+
|                                     INFLUXDB V2 TSM ENGINE                                 |
| Bucket: factory_telemetry | Org: bosch_twin | Retention: 30 days                           |
+────────────────────────────────────────────────────────────────────────────────────────────+
                                             ▲
                                             │ (Flux Queries)
+────────────────────────────────────────────────────────────────────────────────────────────+
|                                      GRAFANA 11                                            |
| Dashboards: Line Overview | Station Detail | Parts & Risk | AI Insights                    |
| Provisioned Unified Alerting: Critical High-Risk Surge | Station Starvation | LSTM Anomaly |
+────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Component Breakdown

### 2.1 Replay Engine (`twin.replay`)
- **Core modules:** [engine.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/replay/engine.py), [run.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/replay/run.py), [build_replay_data.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/replay/build_replay_data.py).
- **Function:** Reconstructs historical part arrivals, inter-station transit pacing, and sensor telemetry across stations S29–S37 in strict chronological sequence.
- **Clock Synchronization:** Multiplied virtual time scale ($16.75\text{ units} = 168\text{ hours}$). Includes gap detection that fast-forwards prolonged idle gaps.

### 2.2 Ingestion Pipeline (`twin.ingest`)
- **Core modules:** [mqtt_to_influx.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/ingest/mqtt_to_influx.py).
- **Function:** Subscribes to wildcard MQTT factory topics (`factory/line3/#`, `twin/ai/#`), unpacks JSON payloads into nanosecond-precision InfluxDB measurement Points, and commits them via synchronous batching.

### 2.3 Real-Time Scoring & Inference (`twin.scoring` & `twin.ai`)
- **Core modules:** [scoring_service.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/scoring/scoring_service.py), [inference.py](file:///c:/projects/HCL_Projects/PROJECT/src/twin/ai/inference.py).
- **Function:**
  - Maintains stateful per-station rolling windows of transit times, defect counts, and sensor readings.
  - Generates multi-factor composite Station Health ($H_s \in [0, 100]$).
  - Evaluates part defect risk at the S34 routing fork before entry to S35/S36 using calibrated PyTorch MLP.
  - Aggregates 1-sim-hour non-gap windows and executes unsupervised 24-step LSTM Autoencoder sequence scoring on GPU.

### 2.4 Data Storage & Visualization (`infra/`)
- **InfluxDB v2:** In-memory index and Time-Structured Merge tree storage engine for rapid Flux time-series aggregations.
- **Grafana 11:** Pre-provisioned multi-tenant dashboard suite with live refresh (1s/5s) and automated alerting.

---

## 3. Data Dictionary & MQTT Topics

| Topic | Payload Schema | InfluxDB Measurement |
| :--- | :--- | :--- |
| `factory/line3/{station_id}/telemetry` | `StationTelemetryEvent` (part_id, sim_time, features) | `station_telemetry` |
| `factory/line3/{station_id}/health` | `StationHealthEvent` (health_score, drift, pacing, defects) | `station_health` |
| `factory/line3/parts/{part_id}/risk` | `PartRiskEvent` (part_id, composite_risk, risk_band) | `part_risk` |
| `factory/line3/overview` | `LineOverviewEvent` (parts_entered, completed, current_wip) | `line_overview` |
| `twin/ai/part_risk` | Defect MLP inference (p_calibrated, p_raw, lead_time) | `ai_part_risk` |
| `twin/ai/anomaly` | LSTM Autoencoder inference (reconstruction_error, contributions) | `ai_sequence_anomaly` |
