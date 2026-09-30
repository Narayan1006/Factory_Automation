# AI-Driven Digital Twin for Smart Factory Operations (v3.0)
**HCL In-House Internship Project**  
*Technology Stack: Python 3.12 | PyTorch (CUDA 12.4 on RTX 3050) | InfluxDB v2 | Eclipse Mosquitto MQTT | Grafana 11*

> An evidence-based, real-time industrial Digital Twin for Line 3 of the Bosch Production Line Performance dataset (1.18M parts). Combines deterministic discrete-event historical replay over MQTT, sub-second time-series ingestion into InfluxDB v2, multi-factor station health scoring, and GPU-accelerated PyTorch deep learning models (calibrated defect MLP and LSTM autoencoder) to provide 30–75 minutes of proactive quality lead time before physical exit inspection.

---

## 1. System Architecture

```
                       +-----------------------------------------------+
                       |          HISTORICAL REPLAY ENGINE             |
                       |  Simulates discrete-event clock (1x - 120x)   |
                       |  Streams real Bosch events across S29 -> S37  |
                       +-----------------------------------------------+
                                              │
                                              ▼ (MQTT: factory/line3/{stn}/telemetry)
       +─────────────────────────────────────────────────────────────────────────────+
       │                                                                             │
       ▼                                                                             ▼
 +──────────────────────────────+                              +──────────────────────────────+
 |   SCORING & AI SERVICE       |                              |   INGESTION SERVICE          |
 | - Multi-factor Station Health|                              | - Subscribes to all topics   |
 | - Rolling Pacing & Drift     |                              | - Converts JSON to Points    |
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
 |                                       GRAFANA 11                                           |
 | Dashboards: Line Overview | Station Detail | Parts & Risk | AI Insights                    |
 | Provisioned Alerts: High-Risk Surge | Station Starvation | LSTM Anomaly | Pipeline Stall   |
 +────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Five-Command Quickstart

```powershell
# 1. Clone repository & install package in editable mode
git clone https://github.com/Narayan1006/Factory_Automation.git
cd Factory_Automation
pip install -e .

# 2. Launch Docker infrastructure (Mosquitto MQTT, InfluxDB v2, Grafana 11)
docker compose -f infra/docker-compose.yml up -d

# 3. Launch end-to-end live demo with automated service orchestration
.\scripts\demo.ps1 -Scenario A -Speed 120.0

# 4. Run test suite (Phase 1 platform, Phase 2 AI, and Path resolution)
pytest -v

# 5. Open Grafana Dashboards (admin / admin_password_123)
Start-Process "http://localhost:3000/d/line3-overview"
```

---

## 3. Project Components & Repository Map

| Component | Path | Description |
| :--- | :--- | :--- |
| **Package Core** | [`src/twin/core/`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/core) | Schemas, ISA-95 MQTT topics, baselines, and health scoring. |
| **Replay Engine** | [`src/twin/replay/`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/replay) | Historical simulation clock, gap detection, and MQTT publisher. |
| **Ingestion Pipeline** | [`src/twin/ingest/`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/ingest) | Wildcard MQTT listener and InfluxDB v2 Line Protocol committer. |
| **Scoring Service** | [`src/twin/scoring/`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/scoring) | Real-time multi-factor station health and part risk tracking. |
| **PyTorch AI Layer** | [`src/twin/ai/`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/ai) | Defect MLP, LSTM Autoencoder, Platt calibration, and inference. |
| **Central Paths** | [`configs/paths.yaml`](file:///c:/projects/HCL_Projects/PROJECT/configs/paths.yaml) | Master filesystem path registry resolved via [`twin.paths`](file:///c:/projects/HCL_Projects/PROJECT/src/twin/paths.py). |
| **Dashboards & Alerts**| [`grafana/`](file:///c:/projects/HCL_Projects/PROJECT/grafana) | 4 JSON dashboards and file-provisioned alerting rules. |
| **Docker Stack** | [`infra/docker-compose.yml`](file:///c:/projects/HCL_Projects/PROJECT/infra/docker-compose.yml) | Multi-container setup for Mosquitto, InfluxDB, and Grafana. |
| **Demo Orchestrator** | [`scripts/demo.ps1`](file:///c:/projects/HCL_Projects/PROJECT/scripts/demo.ps1) | One-click PowerShell runner managing background services. |
| **Figure Generator** | [`scripts/make_figures.py`](file:///c:/projects/HCL_Projects/PROJECT/scripts/make_figures.py) | Generates presentation figures in [`docs/phase3_demo/figures/`](file:///c:/projects/HCL_Projects/PROJECT/docs/phase3_demo/figures). |

---

## 4. Machine Learning & Engineering Performance Summary

| Metric | Empirical Baseline / Target | Digital Twin Result | Operational Impact |
| :--- | :---: | :---: | :--- |
| **Factory Throughput Scope** | Line 3 core sequence | **89.02%** (1,053,742 parts) | Covers dominant manufacturing stream without synthetic routes |
| **Proactive Lead Time** | 0 min (exit QC at S37) | **+30 to 75 minutes** | Evaluated at S34 fork before physical entry to S35/S36 |
| **Defect Precision Lift@1%** | 1.0x (random sampling) | **5.8x to 7.26x** | Inspecting top 1% highest-risk parts captures >7% of all defects |
| **Defect PR-AUC** | 0.0051 (random baseline) | **0.20 – 0.27** | **40x–50x improvement** over random in 0.51% imbalanced data |
| **LSTM Anomaly Specificity** | 95.0% | **99.0%** (p99 threshold) | Unsupervised sequence anomaly flagged without false alarm storms |
| **Replay Throughput** | Real-time (1x) | **>7,000 events/sec** (burst) | Instantaneous simulation testing and regression validation |
| **Causal Integrity** | Zero future leakage | **Strict forward-chaining** | Validated across 4 Leave-Scenario-Out holdout models |

---

## 5. Defense & Demonstration Dossier

- 📋 **[15-Question Defense Sheet](file:///c:/projects/HCL_Projects/PROJECT/docs/phase3_demo/DEFENSE_SHEET.md):** Rigorous answers to the toughest technical defense questions (anonymization, PR-AUC honesty, causal leakage prevention).
- ⏱️ **[Live Demo Presentation Script](file:///c:/projects/HCL_Projects/PROJECT/docs/phase3_demo/DEMO_SCRIPT.md):** Minute-by-minute timeline for Scenarios A, B, C, and D.
- 📐 **[System Architecture](file:///c:/projects/HCL_Projects/PROJECT/docs/architecture.md):** End-to-end technical architecture and data flow.
- 📑 **[Documented Assumptions](file:///c:/projects/HCL_Projects/PROJECT/docs/assumptions.md):** Explicit empirical assumptions (16.75 time unit week, pacing benchmarks).
- ⚠️ **[Project Limitations](file:///c:/projects/HCL_Projects/PROJECT/docs/limitations.md):** Boundary conditions and industrial manufacturing constraints.
- 📜 **[Architecture Decision Records](file:///c:/projects/HCL_Projects/PROJECT/docs/decisions.md):** ADR-001 through ADR-008 documenting all technical design choices.
- 🔔 **[Alerting Runbook](file:///c:/projects/HCL_Projects/PROJECT/docs/phase1_platform/ALERTING.md):** Grafana alert definitions, thresholds, and operational actions.
- 🛣️ **[Evolutionary Roadmap](file:///c:/projects/HCL_Projects/PROJECT/docs/PHASES.md):** Progression from Phase 0 to Phase 3.
