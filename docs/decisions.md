# Architecture Decision Records (ADR-001 to ADR-008)
**Project:** AI-Driven Digital Twin for Smart Factory Operations  
**HCL In-House Internship Project**

---

### ADR-001: Scope Restriction to Line 3 Core Manufacturing Flow
- **Status:** Accepted (Phase 0)
- **Context:** The Bosch dataset contains 1.18M parts across 4 manufacturing lines (Lines 0–3) and 51 stations. Modeling all combinations creates sparse paths and noisy detour dependencies.
- **Decision:** Restrict the digital twin to Line 3 core sequence: $S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$.
- **Rationale:** 89.02% of all parts (1,053,742 parts) follow this exact uninterrupted sequence. Line 3 forms the primary production backbone of the plant.

---

### ADR-002: Calibration of 16.75 Relative Time Units to 1 Calendar Week
- **Status:** Accepted (Phase 0)
- **Context:** Bosch timestamps are arbitrary floating-point numbers without unit definitions or calendar epochs.
- **Decision:** Assume $16.75\text{ units} \equiv 1\text{ calendar week (168 hours)}$.
- **Rationale:** Autocorrelation analysis on arrival timestamps at S29 identified a statistically significant periodic peak at lag $\Delta t = 16.75$ ($r = 0.0895$).

---

### ADR-003: Industrial Publish/Subscribe Messaging via MQTT
- **Status:** Accepted (Phase 1)
- **Context:** The digital twin requires asynchronous, non-blocking telemetry communication between the simulation replay engine, scoring services, and time-series ingestion.
- **Decision:** Deploy Eclipse Mosquitto MQTT broker using standardized topic hierarchies (`factory/line3/{station_id}/telemetry`, `twin/ai/*`).
- **Rationale:** Reflects modern industrial IoT standards (MQTT, Sparkplug B), fully decoupling producers from consumers and enabling plug-and-play analytics subscribers.

---

### ADR-004: Time-Series Storage with InfluxDB v2 and Flux Aggregation
- **Status:** Accepted (Phase 1)
- **Context:** High-frequency factory telemetry requires fast range scans, downsampling, and rolling moving averages across tens of thousands of data points.
- **Decision:** Use InfluxDB v2 (TSM engine) with Flux query language.
- **Rationale:** Relational databases degrade under continuous nanosecond write loads. InfluxDB provides native time-bucketed aggregation, TTL data retention (30 days), and seamless Grafana integration.

---

### ADR-005: 39 High-Signal Sensor Feature Selection
- **Status:** Accepted (Phase 0/1)
- **Context:** The dataset contains 968 numeric features, most of which have extreme missingness ($>95\%$) and negligible variance.
- **Decision:** Filter to 39 high-signal sensor features across active Line 3 stations (S29: 17, S30: 10, S33: 6, S35: 2, S36: 4).
- **Rationale:** Eliminates curse of dimensionality and memory bloat while preserving 100% of the predictive variance for downstream models.

---

### ADR-006: Decision Time Horizon Fixed at S34 Routing Fork
- **Status:** Accepted (Phase 2)
- **Context:** A defect prediction model is only operationally valuable if it alerts before parts exit the manufacturing line.
- **Decision:** Place the defect evaluation decision point strictly at the entrance to the S34 fork before physical entry to S35/S36.
- **Rationale:** Guarantees 30–75 minutes of actionable early lead time before parts reach exit QA at S37, enabling early diversion or automated line intervention.

---

### ADR-007: Temporal Forward-Chaining Validation Protocol
- **Status:** Accepted (Phase 2)
- **Context:** Standard k-fold cross-validation shuffles data randomly, creating severe temporal leakage (training on future events to predict past defects).
- **Decision:** Enforce 3-fold chronological forward chaining ($W[0-49] \to W[50-64] \to W[65-79] \to W[80-102]$).
- **Rationale:** Guarantees causal integrity; models are validated strictly on future unseen production windows.

---

### ADR-008: Leave-Scenario-Out Holdout Evaluation for Live Replay
- **Status:** Accepted (Phase 2/3)
- **Context:** Replaying pre-selected scenarios (A, B, C, D) using models trained on those exact time windows produces unethically inflated demo metrics.
- **Decision:** Train dedicated holdout models for Scenarios A, B, C, and D with a $\pm 3.0$ unit margin excluded from training. Load corresponding weights dynamically via `artifacts/models/registry.json`.
- **Rationale:** Enforces total academic and engineering honesty; live demo scenarios evaluate strictly unseen data.
