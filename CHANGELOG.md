# Changelog: Bosch Digital Twin

All notable changes to this project are documented in this file following the Keep a Changelog format.

## [3.0.0] - 2026-09-30 (Phase 3: Production Refactoring, Alerting & Polish)
### Added
- Standardized package architecture under `src/twin/` with editable install (`pip install -e .`).
- Declarative Grafana Unified Alerting (`grafana/provisioning/alerting/rules.yaml` and `contact_points.yaml`).
- Automated PowerShell demo orchestration runner `scripts/demo.ps1` with scenario switching and background process lifecycle management.
- Comprehensive presentation figures generator `scripts/make_figures.py`.
- Complete project defense sheet `docs/phase3_demo/DEFENSE_SHEET.md` with 15 core technical defense questions.
- Master documentation suite: `docs/architecture.md`, `docs/assumptions.md`, `docs/limitations.md`, `docs/decisions.md` (ADR-001 to ADR-008), and `docs/PHASES.md`.
- Centralized path management (`configs/paths.yaml`, `src/twin/paths.py`).

### Changed
- Refactored all module imports from root namespace to `twin.*`.
- Migrated test suites into modular hierarchy (`tests/phase1_platform/`, `tests/phase2_ai/`, `tests/test_paths.py`).

---

## [2.0.0] - 2026-09-30 (Phase 2: Deep Learning Layer & Causal Evaluation)
### Added
- PyTorch GPU-accelerated Defect Risk MLP model (128 -> 64) with Platt probability calibration.
- PyTorch GPU-accelerated Anomaly Detection LSTM Autoencoder over 24-hour non-gap sequences.
- Forward-chaining temporal evaluation protocol with zero future-outcome leakage.
- Leave-Scenario-Out holdout models for live demo scenarios A, B, C, and D.
- Fourth Grafana dashboard: *Line 3 AI Insights*.
- Technical AI report `docs/phase2_ai/REPORT_AI.md`.

---

## [1.0.0] - 2026-09-30 (Phase 1: Real-Time Platform Architecture)
### Added
- Docker Compose infrastructure: Eclipse Mosquitto, InfluxDB v2, Grafana 11.
- Deterministic SimPy-style historical replay engine with simulation speed multipliers.
- Asynchronous MQTT to InfluxDB v2 ingestion pipeline.
- Multi-factor composite Station Health ($H_s$) and Part Risk ($Q$) real-time scoring engine.
- Three provisioned Grafana dashboards: *Line Overview*, *Station Detail*, *Parts & Risk*.

---

## [0.1.0] - 2026-09-30 (Phase 0: Dataset Exploration & Baseline Discovery)
### Added
- Full statistical exploration of the 1,183,747 parts across 968 numeric features.
- Identification of Line 3 core manufacturing topology ($S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$).
- Empirical baseline extraction: arrival rate, transit pacing benchmarks, 0.51% defect rate.
- Discovery of 16.75 time unit autocorrelation weekly seasonality.
- Comprehensive analytical report `docs/phase0_exploration/DATASET_REPORT.md`.
