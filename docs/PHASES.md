# Project Roadmap: Evolutionary Phases of the Bosch Digital Twin
**Project:** AI-Driven Digital Twin for Smart Factory Operations  
**Institution:** HCL In-House Internship

---

```
  Phase 0: Dataset Exploration & Topology Discovery (v0.1)
  ├── 1.18M parts analyzed; Line 3 identified as 89.02% core manufacturing throughput
  └── 39 high-signal sensor features isolated; 16.75 time unit weekly seasonality uncovered
                           │
                           ▼
  Phase 1: Real-Time Platform & Edge Simulation Engine (v1.0)
  ├── Docker container stack (Mosquitto MQTT broker, InfluxDB v2, Grafana 11)
  ├── Deterministic SimPy-style replay engine streaming historical shop-floor telemetry
  └── Rule-based multi-factor station health scoring and initial dashboards
                           │
                           ▼
  Phase 2: Deep Learning Layer & Causal Evaluation (v2.0)
  ├── Part-level Defect Risk MLP (128 -> 64) with Platt probability calibration
  ├── Line/Station Anomaly Detection LSTM Autoencoder over 24h rolling windows
  └── Forward-chaining temporal evaluation (Lift@1% up to 7.26x) and ablation studies
                           │
                           ▼
  Phase 3: Production Refactoring, Alerting & Defense Polish (v3.0)
  ├── Strict clean-architecture package restructuring (`src/twin/`, `configs/`, `artifacts/`)
  ├── Grafana Unified Alerting with declarative rules and contact points
  └── Automated demo orchestrator, 15-question defense sheet, and QA runbook
```

---

## Phase 0: Exploration & Grounding (v0.1)
- **Objective:** Establish transparent empirical truth from the raw 1.18M-row Bosch Production Line Performance dataset without fabrication.
- **Key Deliverables:**
  - `docs/phase0_exploration/DATASET_REPORT.md`: Comprehensive 8-task analytical report.
  - Identification of Line 3 core sequence ($S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$), handling 89.02% of all parts.
  - Empirical baseline extraction: 0.51% baseline defect rate, arrival dynamics, transit pacing benchmarks.
  - Isolation of 39 informative numeric features across 968 candidate columns.

## Phase 1: Real-Time Platform Architecture (v1.0)
- **Objective:** Build an end-to-end industrial IoT telemetry pipeline.
- **Key Deliverables:**
  - Docker Compose environment with Eclipse Mosquitto, InfluxDB v2, and Grafana.
  - `twin.replay.engine`: Deterministic replay engine streaming historical factory events over MQTT topics at configurable speed (1x to 120x, burst mode).
  - `twin.ingest.mqtt_to_influx`: Asynchronous ingestion daemon committing Line Protocol time-series points.
  - `twin.core.health`: Deterministic composite station health ($H_s$) and cumulative part risk ($Q$) scorer.
  - Three provisioned Grafana dashboards: *Line Overview*, *Station Detail*, *Parts & Risk*.

## Phase 2: PyTorch AI Layer & Causal Evaluation (v2.0)
- **Objective:** Enhance the digital twin with supervised defect prediction and unsupervised anomaly detection on RTX 3050 GPU.
- **Key Deliverables:**
  - `twin.ai.defect_model`: 2-layer MLP predicting defect probability at the S34 fork with Platt probability calibration.
  - `twin.ai.anomaly_model`: 1-layer LSTM Autoencoder modeling 24-hour multivariate window sequences with gap isolation.
  - Causal engineering: Enforced strict forward-chaining temporal validation and Leave-Scenario-Out holdout models.
  - `docs/phase2_ai/REPORT_AI.md`: Honest evaluation documenting PR-AUC (0.20-0.27), Lift@1% (5.8x-7.26x), and permutation importance.
  - Fourth Grafana dashboard: *Line 3 AI Insights*.

## Phase 3: Packaging, Alerting & Presentation Polish (v3.0)
- **Objective:** Transform the research codebase into an enterprise-grade installable Python package with declarative alerting, demo orchestration, and comprehensive defense material.
- **Key Deliverables:**
  - Standard packaging: `pyproject.toml`, modular `requirements/`, centralized `configs/paths.yaml` and `twin.paths`.
  - Provisioned Grafana alerting rules in `grafana/provisioning/alerting/`.
  - Automated demo orchestrator `scripts/demo.ps1` with scenario switching and log isolation.
  - Complete defense dossier: `DEFENSE_SHEET.md`, `DEMO_SCRIPT.md`, and reproducible presentation figures.
