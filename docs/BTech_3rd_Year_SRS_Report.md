# SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
## Knowledge-Driven IoT Fault Diagnosis Assistant: AI-Driven Digital Twin for Smart Factory Operations

**Standard Academic Compliance:** IEEE Std 830-1998  
**Department:** Department of Mechanical Engineering  
**Institution:** ABES Engineering College / Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow  
**Academic Year:** 2026 – 2027  
**Status:** 3rd Year B.Tech Technical Project Specification  

---

### SUBMITTED BY:
- **Candidate:** Nishchay Arora
- **University Roll No:** 2400320400036
- **Degree:** Bachelor of Technology (B.Tech) in Mechanical Engineering

### UNDER THE SUPERVISION OF:
- **Project Guide:** Prof. / Dr. Tanvi Saxena
- **Department:** Department of Mechanical Engineering, ABES EC, Ghaziabad / AKTU, Lucknow
- **Date of Submission:** 30-September-2026

---

### Document Revision & Approval History

| Version | Date | Description / Major Changes | Prepared By | Reviewed By |
| :--- | :---: | :--- | :---: | :---: |
| **0.1** | 15-Sep-2026 | Initial Draft, Problem Definition & Scope Boundary | Nishchay Arora | Dr. Tanvi Saxena |
| **1.0** | 30-Sep-2026 | Final 3rd-Year Mid-Term SRS Baseline (IEEE Std 830-1998) | Nishchay Arora | HOD / Review Committee |

---

## Table of Contents
1. [Introduction](#1-introduction)
   - 1.1 Purpose
   - 1.2 Scope of the System
   - 1.3 Definitions, Acronyms, and Abbreviations
   - 1.4 References
   - 1.5 Document Overview
2. [Overall Description](#2-overall-description)
   - 2.1 Product Perspective & Context Architecture
   - 2.2 User Classes and Characteristics
   - 2.3 Operating Environment
   - 2.4 Design and Implementation Constraints
   - 2.5 Assumptions and Dependencies
3. [System Features and Functional Requirements](#3-system-features-and-functional-requirements)
   - 3.1 Module 1: Ingestion, Streaming & Discrete-Event Replay
   - 3.2 Module 2: Edge Analytics, Multi-Factor Health Scoring & AI Inference
   - 3.3 Module 3: Time-Series Storage, Alerting, Visualization & Control
4. [External Interface Requirements](#4-external-interface-requirements)
   - 4.1 User Interfaces (UI)
   - 4.2 Hardware & Edge IoT Interfaces
   - 4.3 Software Interfaces
   - 4.4 Communication Interfaces
5. [Non-Functional Requirements (NFRs)](#5-non-functional-requirements-nfrs)
   - 5.1 Performance Requirements
   - 5.2 Security & Access Governance
   - 5.3 Reliability & Availability
   - 5.4 Portability, Maintainability & Modularity
6. [System Design and Analysis Models](#6-system-design-and-analysis-models)
   - 6.1 Use Case Modeling & Specification Tables
   - 6.2 Data Flow Diagrams (DFD Level 0 Context & DFD Level 1 Detailed)
   - 6.3 Entity-Relationship & Time-Series Data Schema
   - 6.4 UML Sequence & Activity Dynamic Models
7. [Academic Review & Sign-Off](#7-academic-review--sign-off)

---

## 1. Introduction

### 1.1 Purpose
The purpose of this Software Requirements Specification (SRS) document is to provide a complete, rigorous, and unambiguous formal definition of the software requirements for the **"Knowledge-Driven IoT Fault Diagnosis Assistant: AI-Driven Digital Twin for Smart Factory Operations"**. 

This specification conforms strictly to the **IEEE Std 830-1998 Recommended Practice for Software Requirements Specifications**. It establishes the functional, performance, external interface, and non-functional baselines required for academic evaluation within the B.Tech 3rd-year Mechanical Engineering curriculum, while serving as a binding software engineering contract between the student developer, academic evaluators, and industrial manufacturing stakeholders.

### 1.2 Scope of the System
In high-volume manufacturing environments, unexpected machine downtime, undetected sensor calibration drift, and lagging end-of-line quality control result in severe yield degradation and costly scrap. Traditional supervisory control systems rely on static thresholds and lagging post-manufacturing inspection at the exit gate, forfeiting the opportunity for proactive in-line intervention.

The proposed system addresses this industrial challenge by engineering an end-to-end, real-time **Cyber-Physical Digital Twin** powered by the empirical **Bosch Production Line Performance** dataset (1,183,747 parts across multi-stage production lines).

- **In-Scope Capabilities:**
  1. *Deterministic Shop-Floor Telemetry Streaming:* Industrial IoT publish/subscribe architecture via Eclipse Mosquitto MQTT streaming historical discrete events with configurable simulation speed scaling (1x to 120x, and burst mode >7,000 events/sec).
  2. *Empirical Manufacturing Topology Modeling:* Core sequence tracking across Line 3 ($S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$), representing 89.02% (1,053,742 parts) of plant throughput.
  3. *Multi-Factor Edge Health Scoring:* Real-time stateful computation of composite Station Health ($H_s \in [0, 100]$) combining sensor $|Z|$-score drift ($w=0.45$), transit pacing delay ($w=0.30$), and rolling local failure rates ($w=0.25$).
  4. *Deep Learning Predictive Quality Engine:* GPU-accelerated PyTorch Multilayer Perceptron (MLP) evaluated at the S34 routing fork with Platt probability calibration, providing 30–75 minutes of early lead time prior to exit inspection.
  5. *Unsupervised Sequence Anomaly Detection:* 1-layer PyTorch LSTM Autoencoder tracking 24-hour multivariate station feature vectors to isolate latent multi-station process anomalies before component breakdown.
  6. *Industrial Time-Series Persistence & Unified Alerting:* Sub-second ingestion into InfluxDB v2 and declarative Grafana 11 dashboards with file-provisioned alert notifications.

- **Out-of-Scope Boundaries:**
  1. *Physical Actuator Hardware:* Closed-loop pneumatic PLC diverter actuation or conveyor motor control (open-loop predictive advisory mode is implemented).
  2. *Enterprise ERP Integration:* SAP/Oracle enterprise resource planning billing or procurement modules.
  3. *Unverified Physical Tooling Fabrication:* Physical machine naming or fictionalized failure root causes (stations S29–S37 are modeled strictly by their empirical statistical and topological parameters).

- **Expected Benefits:**
  - **Early Warning Lead Time:** Provides +30 to 75 minutes of actionable defect warning prior to physical exit inspection at S37.
  - **Inspection Efficiency:** Yields a Lift@1% of **5.8x to 7.26x**, enabling plant quality inspectors to detect >7% of all defective parts by sampling merely the top 1% highest-risk assemblies.
  - **Scrap Reduction:** Enables early diversion of compromised batches before irreversible downstream assembly.

### 1.3 Definitions, Acronyms, and Abbreviations

| Category | Term / Acronym | Definition / Standard Academic Context |
| :--- | :--- | :--- |
| **Standard** | **SRS** | Software Requirements Specification (IEEE Std 830-1998 format). |
| **Standard** | **IEEE** | Institute of Electrical and Electronics Engineers. |
| **Architecture**| **Digital Twin** | Virtual software representation of physical factory machines and processes updated in real-time. |
| **Architecture**| **IIoT** | Industrial Internet of Things; interconnected sensors and edge compute devices in manufacturing. |
| **Network** | **MQTT** | Message Queuing Telemetry Transport; lightweight, binary ISO standard pub/sub protocol (ISO/IEC 20922). |
| **Database** | **TSM / Flux** | Time-Structured Merge tree storage engine in InfluxDB v2; Flux functional data scripting language. |
| **Deep Learning**| **MLP** | Multilayer Perceptron; feedforward artificial neural network used for defect risk classification. |
| **Deep Learning**| **LSTM-AE** | Long Short-Term Memory Autoencoder; recurrent neural architecture for temporal reconstruction anomaly scoring. |
| **Statistics** | **Z-Score** | Standard score expressing distance from the empirical mean in standard deviations ($Z = (x - \mu)/\sigma$). |
| **Statistics** | **PR-AUC** | Precision-Recall Area Under Curve; primary evaluation metric for severe class imbalance ($0.51\%$ defect rate). |
| **Calibration** | **Platt Scaling**| Logistic regression transformation applied to raw neural network logits to produce true Bayesian probabilities. |
| **Protocol** | **ISA-95** | International standard for developing automated enterprise-to-control system interfaces. |

### 1.4 References
1. **IEEE Std 830-1998:** *IEEE Recommended Practice for Software Requirements Specifications*, IEEE Computer Society, 1998.
2. **Sommerville, Ian:** *Software Engineering (10th Edition)*, Pearson Education, 2015.
3. **Chi, Y., Dong, Y., Wang, Z. J., Yu, F. R., and Leung, V. C. M.:** "Knowledge-Based Fault Diagnosis in Industrial Internet of Things: A Survey," *IEEE Internet of Things Journal*, vol. 9, no. 15, pp. 12886–12900, 2022.
4. **Bosch Group:** *Bosch Production Line Performance: Anonymized Manufacturing Quality Dataset*, Kaggle Industrial Competition, 2016.
5. **Light, R. A.:** "Mosquitto: An Open Source MQTT Server," *Journal of Open Source Software*, vol. 2, no. 13, 2017.
6. **InfluxData:** *InfluxDB v2 OSS Documentation: Line Protocol & Flux Engine*, InfluxData Inc., 2024.
7. **Paszke, A., et al.:** "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.

### 1.5 Document Overview
The remainder of this specification is organized into structured sections:
- **Section 2 (Overall Description):** High-level product perspective, user personas, operating runtime environments, and documented assumptions.
- **Section 3 (System Features):** Itemized functional requirements (FR-1.1 through FR-3.4) with input validation rules and priorities.
- **Section 4 (External Interfaces):** Specifications for UI, Edge IoT hardware, database connectors, and network communication.
- **Section 5 (Non-Functional Requirements):** Quantitative targets for performance, security, reliability, and maintainability.
- **Section 6 (System Design Models):** Comprehensive academic diagrams including Use Case, DFD Levels 0 & 1, ERD/TSM schemas, and UML Sequence workflows.
- **Section 7 (Academic Review):** Formal evaluation rubrics and departmental sign-off page.

---

## 2. Overall Description

### 2.1 Product Perspective & Context Architecture
The **Knowledge-Driven IoT Fault Diagnosis Assistant** is a self-contained, multi-tier cyber-physical digital twin designed for modern industrial manufacturing operations. It operates in real time above the plant floor level, decoupling physical/simulated telemetry producers from analytical consumers via an asynchronous messaging broker.

```
+────────────────────────────────────────────────────────────────────────────────────────────+
|                                    FACTORY SHOP FLOOR                                      |
|  Sensors: L3_S29_F3315 ... L3_S36_F3840 (39 Informative Industrial Sensor Features)       |
|  Topological Stations: S29 (Entry) -> S30 -> S33 -> S34 (Fork) -> (S35 | S36) -> S37 (Exit)|
+────────────────────────────────────────────────────────────────────────────────────────────+
                                              │
                                              ▼ (JSON Telemetry over MQTT 1883)
+────────────────────────────────────────────────────────────────────────────────────────────+
|                               INDUSTRIAL EDGE IoT GATEWAY                                  |
|  Eclipse Mosquitto Broker (Topic Hierarchy: factory/line3/{station_id}/telemetry)         |
+────────────────────────────────────────────────────────────────────────────────────────────+
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
+──────────────────────────────────────────+   +──────────────────────────────────────────+
|      STATEFUL SCORING & AI ENGINE        |   |       TIME-SERIES INGESTION DAEMON       |
| - Baseline Manager (Z-score normalizer)  |   | - Multi-threaded MQTT Consumer           |
| - Health Scorer (Drift, Pacing, Defect)  |   | - Line Protocol Point Serializer         |
| - PyTorch Defect MLP (S34 Fork Lead Time)|   | - Synchronous Batch Commits (50 pts/sec) |
| - PyTorch LSTM Autoencoder (24h Windows) |   +──────────────────────────────────────────+
+──────────────────────────────────────────+                                 │
                       │ (twin/ai/*, factory/line3/{health, risk})           │
                       └──────────────────────┬──────────────────────────────┘
                                              ▼
+────────────────────────────────────────────────────────────────────────────────────────────+
|                               PERSISTENCE & ANALYTICS LAYER                                |
|  InfluxDB v2 TSM Engine (Bucket: factory_telemetry, Org: bosch_twin, Retention: 30 days)   |
+────────────────────────────────────────────────────────────────────────────────────────────+
                                              │
                                              ▼ (Flux Queries / Unified Alert Rules)
+────────────────────────────────────────────────────────────────────────────────────────────+
|                               PRESENTATION & OPERATIONAL UI                                |
|  Grafana 11 Multi-Panel Dashboards (Line Overview, Station Detail, Parts Risk, AI Insights)|
|  Automated Unified Alerting (Critical Quality Surges, Equipment Starvation, Model Anomaly) |
+────────────────────────────────────────────────────────────────────────────────────────────+
```

### 2.2 User Classes and Characteristics
The application supports four primary user classes across factory operations:

| User Class / Persona | Technical Proficiency | Operational Role & Access Rights |
| :--- | :--- | :--- |
| **Plant Operations Engineer** | High (Industrial / Systems Engg) | Full real-time operational oversight; monitors Line Overview dashboard; assesses station status matrix; tunes transit pacing thresholds. |
| **Quality Control Specialist** | Intermediate (Manufacturing QA) | Focuses on Parts & Risk and AI Insights dashboards; reviews parts flagged $>0.70$ risk; executes root-cause drill-downs into station sensor traces. |
| **Maintenance Technician** | Intermediate (Electro-Mechanical) | Receives automated equipment health alerts ($H_s < 50$); inspects sensor drift panels ($|Z| > 3\sigma$); diagnoses conveyor bottlenecks. |
| **Academic Supervisor / Evaluator**| High (Software / AI Research) | Audits system architecture; reviews model evaluation metrics (PR-AUC, Lift@1%); verifies causal leakage isolation and reproducible demo runs. |

### 2.3 Operating Environment
- **Host Hardware:** Intel Core i5/i7 (x86_64) or compatible, 16 GB RAM minimum.
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) with CUDA 12.4 support (CPU automatic fallback supported).
- **Operating System:** Microsoft Windows 11 / Windows 10 (PowerShell 7+) or Linux (Ubuntu 22.04 LTS).
- **Virtualization:** Docker Desktop 4.28+ with Linux container engine and Intel VT-x enabled.
- **Programming Language & Core Stack:** Python 3.12, PyTorch 2.6.0+cu124, InfluxDB Client 1.50, Paho-MQTT 2.1, Grafana 11.1.0.

### 2.4 Design and Implementation Constraints
1. **Academic Budget & Tooling:** 100% open-source, reproducible technology stack (Mosquitto, InfluxDB v2 OSS, Grafana OSS, PyTorch) operating locally without recurring cloud fees.
2. **Causal Integrity Constraint:** Strict enforcement of zero future-outcome leakage. The defect model must predict strictly at the S34 fork before physical entry to S35/S36; rolling defect rate features must strictly use parts that completed exit prior to the current part's arrival.
3. **Anonymization Compliance:** Sensor features and machine identifiers remain strictly grounded in original anonymized naming conventions ($S29\text{--}S37$, $L3\_S29\_F3315$). Fictitious physical equipment names are strictly prohibited.
4. **Reproducible Demo Constraint:** Leave-Scenario-Out holdout models are trained with a $\pm 3.0$ unit safety margin around scenarios A, B, C, and D, guaranteeing that live demonstration replaying never tests on training data.

### 2.5 Assumptions and Dependencies
- **Calendar Time Scale Assumption:** 16.75 relative time units in the Bosch dataset correspond to 1 calendar week (168 hours), derived from empirical autocorrelation lag analysis ($r = 0.0895$). Thus, $1.0\text{ unit} \approx 10.03\text{ hours}$ and $0.01\text{ unit} \approx 6.02\text{ minutes}$.
- **Topological Sequence Assumption:** The core sequence $S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$ accurately represents high-volume factory flow (accounting for 89.02% of total plant volume).
- **Network Dependency:** Local loopback networking (`localhost:1883`, `localhost:8086`, `localhost:3000`) available without port collision.

---

## 3. System Features and Functional Requirements

The functional requirements are partitioned into three core software modules. Each requirement is assigned a unique identifier (`FR-x.x`), validation rule, and priority level.

### 3.1 Module 1: Ingestion, Streaming & Discrete-Event Replay

| Req ID | Feature Description | Input / Validation Rule | Priority |
| :--- | :--- | :--- | :---: |
| **FR-1.1** | **Deterministic Historical Replay** | Replays real historical events chronologically from pre-compiled Parquet datasets (`scenario_{A,B,C,D}.parquet`). | **High** |
| **FR-1.2** | **Simulation Speed Multiplier** | Replay speed $S \in [0.0, 3600.0]$. Default $120.0\times$ (1 sim-hour in 30s); $0.0$ activates burst execution ($>7,000$ events/sec). | **High** |
| **FR-1.3** | **Production Gap Fast-Forwarding**| Detects idle production stoppages ($\Delta t > 0.02$ units / 12 minutes); fast-forwards gap periods in 5.0s wall-clock time. | **Medium** |
| **FR-1.4** | **ISA-95 Topic Serialization** | Publishes JSON payloads conforming to `StationTelemetryEvent` to `factory/line3/{station_id}/telemetry` via MQTT QoS 0/1. | **High** |

### 3.2 Module 2: Edge Analytics, Multi-Factor Health Scoring & AI Inference

| Req ID | Feature Description | Input / Validation Rule | Priority |
| :--- | :--- | :--- | :---: |
| **FR-2.1** | **Empirical Baseline Normalization**| Loads 39 pre-filtered sensor feature distributions ($\mu, \sigma$); computes standard $Z$-scores ($Z = (x - \mu)/\sigma$). | **High** |
| **FR-2.2** | **Multi-Factor Station Health Scoring**| Stateful computation of composite health $H_s = 100 - (0.45 S_{\text{drift}} + 0.30 S_{\text{pacing}} + 0.25 S_{\text{defect}})$. Bounded in $[0, 100]$. | **High** |
| **FR-2.3** | **PyTorch Defect Risk Prediction** | 2-layer MLP ($128 \to 64$) evaluated at S34 routing fork using 85 input features; produces calibrated Bayesian defect probability. | **High** |
| **FR-2.4** | **Platt Probability Calibration** | Fits logistic transformation on validation logits ($P(y=1\|z) = 1 / (1 + e^{Az+B})$); aligns model confidence with 0.51% base rate. | **High** |
| **FR-2.5** | **LSTM Autoencoder Sequence Scoring**| 1-layer LSTM autoencoder evaluates 24 consecutive 1-sim-hour non-gap windows; triggers anomaly when reconstruction error $> p99$. | **Medium** |

### 3.3 Module 3: Time-Series Storage, Alerting, Visualization & Control

| Req ID | Feature Description | Input / Validation Rule | Priority |
| :--- | :--- | :--- | :---: |
| **FR-3.1** | **Asynchronous InfluxDB Commits** | Consumes all telemetry, health, and AI topics; commits Line Protocol points with nanosecond precision in batches of 50 points. | **High** |
| **FR-3.2** | **Multi-Tenant Grafana Dashboards**| Automatically provisions 4 interactive dashboards: *Line Overview*, *Station Detail*, *Parts & Risk*, and *AI Insights*. | **High** |
| **FR-3.3** | **Declarative Unified Alerting** | File-provisioned alert rules in `rules.yaml` monitoring high-risk surges, station starvation, LSTM loss, and pipeline staleness. | **High** |
| **FR-3.4** | **One-Click Demo Orchestration** | PowerShell script (`scripts/demo.ps1`) manages Docker stack, background daemons, scenario selection, and clean termination. | **Medium** |

---

## 4. External Interface Requirements

### 4.1 User Interfaces (UI)
The digital twin presentation layer is delivered through pre-configured Grafana 11 dashboards accessible via standard desktop and mobile browsers at `http://localhost:3000`:
1. **Line 3 Overview Dashboard (`/d/line3-overview`):** High-level plant floor summary featuring parts entered, parts completed, active Work-In-Progress (WIP), rolling defect rate, and real-time Station Health Status Matrix (Green $\ge 80$, Yellow $50\text{--}80$, Red $<50$).
2. **Station Deep-Dive Dashboard (`/d/line3-station-detail`):** Engineering inspection panel with station selector dropdown ($S29\text{--}S37$), live sensor time-series traces against $\pm 3\sigma$ baseline corridors, and inter-station transit pacing percentiles ($p_{90}, p_{99}$).
3. **Parts & Risk Tracking Dashboard (`/d/line3-parts-risk`):** Quality assurance view displaying live part risk distributions, high-risk part inspection queues, and exit QC validation outcomes.
4. **AI Insights Dashboard (`/d/line3-ai-insights`):** Advanced diagnostic panel displaying calibrated defect probabilities ($P_{\text{cal}}$), LSTM autoencoder reconstruction error curves, and station-level error contribution breakdown radar charts.

### 4.2 Hardware & Edge IoT Interfaces
- **Edge Gateway Interface:** The system interfaces with shop-floor data concentrators and simulation nodes over TCP/IP port `1883` using the Eclipse Mosquitto message broker.
- **Compute Interface:** Host workstation leverages NVIDIA Ampere GPU architecture (RTX 3050 Laptop GPU) via CUDA Driver 12.4 and PyTorch Tensor API for sub-15ms neural inference.

### 4.3 Software Interfaces
- **Time-Series Engine Connector:** InfluxDB Client Python API (`influxdb-client 1.50.0`) communicating with InfluxDB v2 over HTTP port `8086` using bearer token authentication.
- **Visualization Engine Connector:** Grafana provisioning engine mapping datasources and dashboards via YAML/JSON specifications stored in `grafana/provisioning/`.
- **Operating System Shell:** Windows PowerShell 5.1/7.x and POSIX Bash compatibility for daemon execution and automated testing.

### 4.4 Communication Interfaces
- **MQTT Protocol (ISO/IEC 20922):** Version 3.1.1 / 5.0 compliant, binary payload transmission over TCP Port `1883`. Keep-alive interval set to 60 seconds.
- **HTTP / HTTPS Protocol:** RESTful API communication and Grafana web interface over TCP Port `3000` and InfluxDB API over TCP Port `8086`.
- **Data Serialization:** Telemetry exchanges structured using strict UTF-8 encoded JSON objects and InfluxDB Line Protocol syntax.

---

## 5. Non-Functional Requirements (NFRs)

| NFR ID | Quality Attribute | Quantitative Target / Standard | Verification & Validation Method |
| :--- | :--- | :--- | :--- |
| **NFR-1** | **Performance (Throughput)** | Replay engine must sustain $\ge 5,000$ events/sec in burst mode without message dropping. | Benchmarked with `run.py --speed 0`: achieved **7,405 events/sec**. |
| **NFR-2** | **Performance (Latency)** | Real-time neural inference latency $< 25$ ms per part on GPU. | Benchmarked on RTX 3050 GPU: achieved **11.4 ms** average batch inference. |
| **NFR-3** | **Security & Access** | Zero plain-text credentials; InfluxDB token-based API access; Grafana RBAC authentication. | Verified in `twin_config.yaml` and Docker container environment isolation. |
| **NFR-4** | **Reliability & Availability** | Continuous streaming uptime $\ge 99.0\%$; zero memory leakage across 24-hour simulation windows. | Deque-bounded sliding windows (`maxlen=10000`); pytest leak verification. |
| **NFR-5** | **ML Model Precision Lift** | Defect model Lift@1% $\ge 5.0\times$ over random baseline in temporal forward chaining. | Verified in `REPORT_AI.md`: achieved **5.8x to 7.26x Lift@1%**. |
| **NFR-6** | **Portability** | Multi-container Docker deployment; installable Python wheel (`pip install -e .`). | Verified via clean-environment runbook (`docs/phase3_demo/RUNBOOK.md`). |

---

## 6. System Design and Analysis Models

### 6.1 Use Case Modeling

#### Use Case Diagram (Textual UML)
```
  +─────────────────────────────────────────────────────────────────────────────+
  |                   Bosch Digital Twin Subsystem Boundary                     |
  |                                                                             |
  |   (UC-01: Launch Replay & Set Speed) <────────── [Plant Operations Eng.]    |
  |   (UC-02: Monitor Live Line Overview) <───────── [Plant Operations Eng.]    |
  |                                                                             |
  |   (UC-03: Inspect Station Health & Drift) <───── [Maintenance Tech.]        |
  |   (UC-04: Receive Predictive Alerts) <────────── [Maintenance Tech.]        |
  |                                                                             |
  |   (UC-05: Query High-Risk Part Queue) <───────── [Quality Specialist]       |
  |   (UC-06: Analyze AI Feature Attribution) <───── [Quality Specialist]       |
  |                                                                             |
  |   (UC-07: Retrain & Validate AI Models) <─────── [Academic Evaluator / ML]  |
  |   (UC-08: Execute Automated Test Suite) <─────── [Academic Evaluator / ML]  |
  +─────────────────────────────────────────────────────────────────────────────+
```

#### Detailed Use Case Specification: UC-05 (Predictive Defect Risk Assessment)
- **Use Case ID:** UC-05
- **Primary Actor:** Quality Control Specialist
- **Pre-Conditions:** Replay engine active; scoring service running with `--ai` enabled; part reaches Station S34 fork.
- **Main Success Scenario:**
  1. Part $P_i$ completes processing at Station S33 and enters the S34 routing fork.
  2. `twin.scoring` captures entry timestamp and computes upstream transit times ($S29 \to S30 \to S33 \to S34$).
  3. Feature vector $X_i \in \mathbb{R}^{85}$ is assembled using S29, S30, S33 sensor readings, throughput, and causal `recent_defect_rate`.
  4. PyTorch Defect MLP executes forward pass; Platt calibrator outputs posterior defect probability $P_{\text{cal}}$.
  5. If $P_{\text{cal}} \ge 0.70$, event is published to `twin/ai/part_risk` and flagged in Grafana High-Risk Queue.
  6. Quality inspector is alerted +30 to 75 minutes prior to part arrival at S37, enabling physical diversion.
- **Post-Conditions:** InfluxDB commits prediction point; exit QC outcome recorded at S37 for post-hoc validation.

---

### 6.2 Data Flow Diagrams (DFD)

#### DFD Level 0: System Context Diagram
```
                     +──────────────────────────────────────+
                     |        Historical Bosch Data         |
                     |  (Parquet Scenarios A, B, C, D)      |
                     +──────────────────────────────────────+
                                        │
                                        │ Raw Chronological Events
                                        ▼
+──────────────+              +────────────────────+              +──────────────+
|  Shop Floor  |  Telemetry   |   LINE 3 DIGITAL   |  Dashboards  |  Plant Ops   |
|   Sensors    | ───────────> |    TWIN SYSTEM     | ───────────> |  & Quality   |
|  (Line 3)    |   (MQTT)     |   (Core Engine)    |   (HTTP)     |  Engineers   |
+──────────────+              +────────────────────+              +──────────────+
                                        │
                                        │ Critical Anomaly Alerts
                                        ▼
                              +────────────────────+
                              |   Maintenance &    |
                              |  Operations Teams  |
                              +────────────────────+
```

#### DFD Level 1: Subsystem Modular Decomposition
```
  [Historical Parquet] ──> (1.0 Replay Engine) 
                                │
                                ▼ (factory/line3/+/telemetry)
                     ┌──────────┴────────────────────────┐
                     │                                   │
                     ▼                                   ▼
        (2.0 Ingestion Pipeline)             (3.0 Scoring & AI Engine)
                     │                                   │
                     │ Line Protocol Points              ├─> 3.1 Z-Score Baseline Normalizer
                     ▼                                   ├─> 3.2 Health Scoring Algorithm
            [ InfluxDB v2 TSM ] <────────────────────────┼─> 3.3 PyTorch Defect MLP (S34 Fork)
                     │                                   └─> 3.4 LSTM Autoencoder (24h Windows)
                     ▼ Flux Queries                              │ (twin/ai/*)
        (4.0 Grafana Visualization) <────────────────────────────┘
                     │
                     ▼
          [ Operational Dashboards ] ──> (5.0 Alerting & Notifications)
```

---

### 6.3 Entity-Relationship & Time-Series Data Schema

Unlike pure relational systems, high-frequency IoT digital twins leverage **Time-Series Schemas** with primary indexing on timestamp, indexed tag dimensions, and numeric field values.

```
+─────────────────────────────────+        +─────────────────────────────────+
|   STATION_TELEMETRY (Measurement)  |        |      STATION_HEALTH (Measurement) |
+─────────────────────────────────+        +─────────────────────────────────+
| Tag: station_id (S29..S37)      |        | Tag: station_id (S29..S37)      |
| Tag: line_id ("3")              |        | Tag: status ("HEALTHY/WARN/CRIT")|
| Field: sim_time (Float)         |        | Field: health_score (0.0-100.0) |
| Field: transit_minutes (Float)  |        | Field: feature_drift_score      |
| Field: L3_S29_F3315..F3840      |        | Field: transit_pacing_score     |
| Time: timestamp_ns (Primary Key)|        | Field: defect_rate_score        |
+─────────────────────────────────+        | Time: timestamp_ns (Primary Key)|
                 │                         +─────────────────────────────────+
                 │ 1:N Part Visits                          ▲
                 ▼                                          │ 1:1 Computed
+─────────────────────────────────+                         │
|       PART_RISK (Measurement)   |─────────────────────────┘
+─────────────────────────────────+
| Tag: risk_band ("LOW/MED/HIGH") |
| Tag: branch ("S35" | "S36")     |
| Field: part_id (Integer)        |
| Field: composite_risk (0.0-1.0) |
| Field: ai_defect_prob (0.0-1.0) |
| Field: ground_truth_label (0/1) |
| Time: timestamp_ns (Primary Key)|
+─────────────────────────────────+
```

---

### 6.4 UML Sequence & Activity Dynamic Models

#### Sequence Diagram: End-to-End Predictive Quality Workflow
```
ReplayEngine           Mosquitto Broker        ScoringService        InfluxDB v2          Grafana UI
     │                        │                      │                    │                   │
     │── 1. Telemetry Event ─>│                      │                    │                   │
     │   (Part at S34 Fork)   │                      │                    │                   │
     │                        │── 2. Route Event ───>│                    │                   │
     │                        │                      │                    │                   │
     │                        │                      │── 3. Assemble X ───│                   │
     │                        │                      │   (Sensors, Drift) │                   │
     │                        │                      │── 4. PyTorch MLP ──│                   │
     │                        │                      │   (P_cal = 0.82)   │                   │
     │                        │                      │                    │                   │
     │                        │<─ 5. Pub AI Risk ────│                    │                   │
     │                        │   (twin/ai/part_risk)│                    │                   │
     │                        │                      │                    │                   │
     │                        │── 6. Ingest Event ───────────────────────>│                   │
     │                        │   (Commit Line Protocol Point)            │                   │
     │                        │                      │                    │                   │
     │                        │                      │                    │── 7. Flux Query ─>│
     │                        │                      │                    │   (Auto-Refresh)  │
     │                        │                      │                    │                   │── 8. Render Risk Alert!
     │                        │                      │                    │                   │   (High Risk Part Queue)
```

---

## 7. Academic Review & Sign-Off

This **Software Requirements Specification (SRS)** has been submitted for academic examination and approved as the technical baseline for the B.Tech 3rd-year engineering project:

- **Project Title:** Knowledge-Driven IoT Fault Diagnosis Assistant: AI-Driven Digital Twin for Smart Factory Operations
- **Student Candidate:** Nishchay Arora (Roll No: `2400320400036`)
- **Degree:** Bachelor of Technology (B.Tech) in Mechanical Engineering
- **Institution:** ABES Engineering College, Ghaziabad / AKTU, Lucknow

### Supervisory Approval Panel:

\
**Project Supervisor / Guide:**  
Prof. / Dr. Tanvi Saxena  
Department of Mechanical Engineering, ABES EC  
Signature: ___________________________  
Date: _______________________________  

\
**Internal Examiner:**  
Department Evaluation Committee  
Signature: ___________________________  
Date: _______________________________  

\
**Head of Department (HOD):**  
Department of Mechanical Engineering  
Signature: ___________________________  
Date: _______________________________  
