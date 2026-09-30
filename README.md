# AI-Driven Digital Twin for Smart Factory Operations (Phase 1)
**HCL In-House Internship Project**  
*Technology Stack: Python 3.12 | MQTT (Eclipse Mosquitto) | Time-Series (InfluxDB v2) | Visualization (Grafana) | Real-Data Replay Engine*

---

## 1. System Overview & Architecture

This project implements an **evidence-based, real-time Digital Twin** for an industrial manufacturing facility, modeled on the **Bosch Production Line Performance** dataset. 

Phase 1 establishes the operational core:
```
                                 [ Bosch Real-Data Replay Engine ]
                                                │
                                    (MQTT: factory/line3/+/telemetry)
                                                ▼
     ┌─────────────────────────────────── Mosquitto MQTT ───────────────────────────────────┐
     │                                     (Port 1883)                                      │
     └──────────────┬───────────────────────────┬───────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
       [ Real-Time Scoring Service ]   [ MQTT-to-Influx Ingestion ]
         - Multi-factor Station Health    - Time-series mapping (ns precision)
         - Part Cumulative Risk           - Tagging & field structuring
         - Throughput & Defect Tracking         │
                    │                           ▼
      (MQTT: .../health, .../risk)        [ InfluxDB v2 ]
                    │                       (Port 8086)
                    └───────────────────────────┤
                                                ▼
                                       [ Grafana Dashboards ]
                                            (Port 3000)
                               - Line 3 Operational Overview
                               - Station Deep Dive & Sensor Traces
                               - Part Risk Tracking & Quality Validation
```

---

## 2. Core Ground Truths & Documented Assumptions

*Crucial for Defense / Presentation:*
1. **Physical Nomenclature:** Anonymized factory identifiers are strictly preserved as $\mathbf{S\langle number\rangle}$ (no fabricated machine names or synthetic sensor roles).
2. **Line 3 Core Focus:** The twin focuses on the dominant sequence:
   $$\mathbf{S29 \longrightarrow S30 \longrightarrow S33 \longrightarrow S34 \longrightarrow (S35 \text{ or } S36) \longrightarrow S37}$$
   - Covers **1,053,742 parts** (**$89.02\%$** of all factory parts).
   - $\text{S35}$ and $\text{S36}$ operate as parallel branches ($49.24\%$ vs $50.76\%$ split).
3. **Temporal Scale Assumption:** Timestamp values are relative units. Based on arrival periodicity analysis ($\text{Lag} = 16.75 \text{ units}$, $r = 0.0895$), we assume $16.75 \text{ units} \equiv 1 \text{ calendar week}$ ($168 \text{ hours}$):
   - $1.0 \text{ unit} \approx 10.03 \text{ hours}$ ($601.8 \text{ minutes}$).
   - $\Delta t = 0.01 \text{ units} \approx 6.02 \text{ minutes}$.
4. **Pacing vs Station Duration:** All date columns within a station share identical timestamps. Therefore, station internal cycle time is zero-delta; line pacing is measured via **inter-station transit time** ($\text{entry}_{i+1} - \text{entry}_i$).
5. **Quality Outcome (`Response`):** The supervisory defect label ($0 = \text{pass}, 1 = \text{fail}$) is a **part-level exit outcome** recorded at $\text{S37}$. Station failure rate strictly denotes the *"failure rate of visiting parts"*, never local machine causation.
6. **No Synthetic Injection:** All anomalies and transients are 100% extracted from the empirical dataset across 4 realistic scenarios.

---

## 3. Directory Layout

```text
├── config/
│   └── twin_config.yaml               # Master configuration, weights, and baselines
├── infra/
│   ├── docker-compose.yml             # Mosquitto, InfluxDB v2, Grafana services
│   ├── .env.example                   # Default container environment variables
│   └── mosquitto/
│       └── mosquitto.conf             # MQTT broker configuration (anonymous access, persistence)
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/influxdb.yml   # Auto-provisioned InfluxDB v2 datasource
│   │   └── dashboards/dashboards.yml  # Auto-provisioned dashboard provider
│   └── dashboards/
│       ├── line_overview.json         # Executive KPI, Line Health, Station Status Matrix
│       ├── station_detail.json        # Sensor traces, transit pacing, station health breakdown
│       └── parts_and_risk.json        # Part risk distribution, high-risk queue, QC validation
├── twin_core/
│   ├── __init__.py                    # Public exports
│   ├── schemas.py                     # Dataclasses & JSON serialization for twin events
│   ├── topics.py                      # ISA-95 standard MQTT topic hierarchy
│   ├── baselines.py                   # Empirical distributions (39 features, transit percentiles)
│   ├── windows.py                     # Rolling deque trackers for moving analytics
│   └── health.py                      # Multi-factor station health & part risk engines
├── replay/
│   ├── build_replay_data.py           # Offline extractor from raw parquet & train_numeric.csv
│   ├── engine.py                      # Real-time event replay streamer with speed scaling
│   └── run.py                         # CLI entrypoint for launching replay scenarios
├── replay_data/                       # Pre-compiled chronological scenario datasets (.parquet)
│   ├── scenario_A.parquet             # t=362-386 (S29 sensor drift & defect burst)
│   ├── scenario_B.parquet             # t=492-502 (~50h stoppage gap & restart surge)
│   ├── scenario_C.parquet             # t=730-745 (Week 44 major defect surge)
│   └── scenario_D.parquet             # t=850-890 (Week 51 shutdown & Week 52 thermal restart)
├── ingest/
│   └── mqtt_to_influx.py              # MQTT consumer -> InfluxDB v2 writer
├── scoring/
│   └── scoring_service.py             # Real-time health & risk computation service
└── tests/
    └── test_twin_pipeline.py          # Automated integration test suite
```

---

## 4. Replay Scenarios (Real Historical Events)

| Scenario | Sim Window ($t$) | Real Duration | Description & Signatures |
| :--- | :---: | :---: | :--- |
| **A** | `362.0 - 386.0` | ~240 hours | S29 feature drift co-occurring with elevated defects ($t \sim 372\text{--}378$). |
| **B** | `492.0 - 502.0` | ~100 hours | $\sim 50\text{h}$ line stoppage ($t=494\text{--}499$, 0 parts logged) followed by a $4.19\%$ defect surge at $t=499$. |
| **C** | `730.0 - 745.0` | ~150 hours | High-density line-wide quality failure surge across multiple production days. |
| **D** | `850.0 - 890.0` | ~400 hours | Factory shutdown in Week 51 (4 parts total), restarting in Week 52 with elevated defects ($1.57\%$). |

---

## 5. How to Run (PowerShell)

### Step 1: Start Container Infrastructure
*(Requires Docker Desktop running with Intel VT-x enabled in BIOS)*
```powershell
docker compose -f infra/docker-compose.yml up -d
```
Verify containers:
```powershell
docker ps
```
- InfluxDB UI: [http://localhost:8086](http://localhost:8086) (User: `admin`, Pass: `admin_password_123`)
- Grafana UI: [http://localhost:3000](http://localhost:3000) (User: `admin`, Pass: `admin_password_123`)

### Step 2: Launch Ingestion Service (Terminal 1)
```powershell
& ".\.venv\Scripts\python.exe" -m ingest.mqtt_to_influx
```

### Step 3: Launch Health & Risk Scoring Service (Terminal 2)
```powershell
& ".\.venv\Scripts\python.exe" -m scoring.scoring_service
```

### Step 4: Stream a Replay Scenario (Terminal 3)
```powershell
# Replay Scenario A at 120x speed (1 simulated hour = 30 wall-clock seconds)
& ".\.venv\Scripts\python.exe" -m replay.run --scenario A --speed 120

# Replay Scenario B in burst mode (immediate ingestion for testing)
& ".\.venv\Scripts\python.exe" -m replay.run --scenario B --speed 0
```

### Step 5: Run Unit & Pipeline Tests
```powershell
& ".\.venv\Scripts\python.exe" -m unittest tests/test_twin_pipeline.py -v
```

---

## 6. Health & Risk Scoring Formulae

### Station Health Index ($H_s \in [0, 100]$)
$$H_s = w_{\text{drift}} S_{\text{drift}} + w_{\text{pacing}} S_{\text{pacing}} + w_{\text{defect}} S_{\text{defect}}$$
- **Feature Drift ($w=0.45$):** Evaluates sensor $|Z|$-scores ($Z = (x - \mu)/\sigma$) for informative features. $|Z| > 3$ triggers anomaly flags. Neutral (100) for S34 and S37 where no informative features exist.
- **Transit Pacing ($w=0.30$):** Compares inter-station delay to empirical $p_{90}$ and $p_{99}$ percentiles. Transit $> p_{99}$ triggers line pacing penalties.
- **Defect Ratio ($w=0.25$):** Compares rolling failure rate of visiting parts to expected baseline ($E = 0.507\%$). $O/E > 3.0$ triggers critical alerts.
- **Thresholds:** Healthy $\ge 80$, Warning $50\text{--}80$, Critical $< 50$.
