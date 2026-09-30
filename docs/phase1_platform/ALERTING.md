# Phase 1 Platform: Grafana Alerting System

## 1. Overview
The Bosch Line 3 Digital Twin alerting subsystem is provisioned declaratively via Grafana Unified Alerting (`grafana/provisioning/alerting/`).
Alerts monitor streaming factory conditions in real time, detecting quality degradation, equipment starvation, unsupervised sequence anomalies, and data pipeline staleness.

---

## 2. Provisioned Alert Rules

### Alert 1: High Risk Part Surge (`alert-high-risk-parts`)
- **Query Measurement:** `part_risk`
- **Field:** `composite_risk`
- **Evaluation Window:** 5 minutes (`-5m`)
- **Threshold Condition:** $> 5$ parts with `composite_risk >= 0.70`
- **Severity:** `critical`
- **Operational Meaning:** An abnormal cluster of parts is traversing Line 3 with severe cumulative defect probability.
- **Action Runbook:**
  1. Inspect the **Line 3 Parts & Risk** dashboard (`/d/line3-parts-risk`).
  2. Identify the active contributing stations (e.g., S29 drift or S33 pacing bottleneck).
  3. Flag downstream QA station S37 for 100% manual sample inspection.

### Alert 2: Station Prolonged Idle / Starvation (`alert-station-idle-warning`)
- **Query Measurement:** `station_telemetry`
- **Evaluation Window:** 5 minutes (`-5m`)
- **Threshold Condition:** Event count $< 1$ across all stations for $\ge 2$ consecutive minutes.
- **Severity:** `warning`
- **Operational Meaning:** Upstream feeding line stoppage or internal conveyor blockage (e.g., matching Scenario B 50-hour gap).
- **Action Runbook:**
  1. Verify if planned shutdown is in effect (e.g., Scenario D holiday window).
  2. Inspect S29 entry buffer; check if parts are accumulating before the entry station.
  3. Prepare thermal pre-heat protocol on S35/S36 before line restart to mitigate cold-start defect bursts.

### Alert 3: AI LSTM Sequence Anomaly (`alert-ai-lstm-anomaly`)
- **Query Measurement:** `ai_sequence_anomaly`
- **Field:** `is_anomaly`
- **Evaluation Window:** 5 minutes (`-5m`)
- **Threshold Condition:** `max(is_anomaly) > 0` (reconstruction loss $> \text{threshold}_{p99}$)
- **Severity:** `critical`
- **Operational Meaning:** The 24-hour sequence of multivariate station metrics deviates significantly from nominal factory operation.
- **Action Runbook:**
  1. Open the **Line 3 AI Insights** dashboard (`/d/line3-ai-insights`).
  2. Inspect the **Station Error Contribution Breakdown** radar/bar panel.
  3. Isolate the specific station with dominant contribution ($> 30\%$).

### Alert 4: Replay Pipeline Stalled (`alert-replay-stalled`)
- **Query Measurement:** `station_health`
- **Evaluation Window:** 2 minutes (`-2m`)
- **Threshold Condition:** Event count $< 1$ for $\ge 2$ minutes while replay is intended to be active.
- **Severity:** `warning`
- **Operational Meaning:** MQTT broker disconnect, scoring service crash, or replay script termination.
- **Action Runbook:**
  1. Check terminal running `twin-scoring` or `twin-replay`.
  2. Check Mosquitto broker container status: `docker compose ps`.
  3. Restart ingestion daemon: `python -m twin.ingest.mqtt_to_influx`.

---

## 3. Notification Routing
Alert notifications are routed via the default contact point `factory-operations-team` defined in `grafana/provisioning/alerting/contact_points.yaml`. For local development, notifications are dispatched to an HTTP webhook endpoint (`http://localhost:8080/alerts`) or standard Grafana server logs.
