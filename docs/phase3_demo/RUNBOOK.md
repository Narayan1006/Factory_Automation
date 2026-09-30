# Cold-Start QA Runbook: Bosch Line 3 Digital Twin

This runbook guides engineers and evaluators through a clean cold-start deployment of the digital twin from an empty or fresh environment.

---

## 1. Prerequisites Checklist
- **Operating System:** Windows 10/11 (or Linux / macOS with Docker).
- **Python:** Python 3.10+ (tested on Python 3.12).
- **Docker Desktop:** Installed and running (ensure Virtualization / VT-x is enabled in BIOS).
- **Hardware (Optional):** NVIDIA GPU with CUDA support for accelerated model retraining (CPU fallback is automatically supported).

---

## 2. Step-by-Step Cold-Start Procedure

### Step 1: Environment Initialization
```powershell
# Clone or navigate to the repository
cd C:\projects\HCL_Projects\PROJECT

# Create virtual environment if missing
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dependencies in editable mode
pip install -e .
```

### Step 2: Launch Containerized Infrastructure
```powershell
docker compose -f infra/docker-compose.yml up -d
```
Verify all 3 containers are healthy:
```powershell
docker compose -f infra/docker-compose.yml ps
```
*Expected output: `twin_mosquitto` (Up, 1883), `twin_influxdb` (Up, 8086), `twin_grafana` (Up, 3000).*

### Step 3: Run Automated Test Suites
Execute the full test suite covering platform pipelines, AI models, and path resolution:
```powershell
pytest -v
```
*Expected result: 15 passed in ~5-6 seconds with zero failures.*

### Step 4: Execute Live Demo Orchestrator
To demonstrate the default scenario (Scenario A: S29 sensor drift & defect burst at 120x speed):
```powershell
.\scripts\demo.ps1 -Scenario A -Speed 120.0
```

To test Scenario B (50-hour stoppage and restart burst):
```powershell
.\scripts\demo.ps1 -Scenario B -Speed 120.0
```

### Step 5: Verify Grafana Dashboards
Open [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin_password_123`):
1. **Line Overview:** `http://localhost:3000/d/line3-overview`
   - Verify parts entered, parts completed, and WIP counters increment live.
   - Verify Station Status Matrix displays green/yellow/red status.
2. **Station Detail:** `http://localhost:3000/d/line3-station-detail`
   - Select Station `S29`. Verify sensor traces `L3_S29_F3315` and transit pacing plots update.
3. **Parts & Risk:** `http://localhost:3000/d/line3-parts-risk`
   - Verify composite part risk points are plotted, and high-risk parts appear in the inspection table.
4. **AI Insights:** `http://localhost:3000/d/line3-ai-insights`
   - Verify calibrated defect probabilities and LSTM reconstruction error loss curves.

---

## 3. Troubleshooting & Common Edge Cases

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| **Port 1883 / 8086 / 3000 in use** | Stale local service or another container | Stop conflicting service (`netstat -ano \| findstr 1883`) or restart Docker Desktop. |
| **MQTT Connection Refused** | Mosquitto container starting slowly | Run `docker compose -f infra/docker-compose.yml restart mosquitto` and retry. |
| **PyTorch CUDA not available** | System missing NVIDIA drivers | System automatically falls back to CPU without error; models run identically. |
| **Replay Parquet missing** | First-time clone without pre-built replay data | Run `python -m twin.replay.build_replay_data --scenario ALL` to regenerate. |
| **InfluxDB unauthorized (401)** | Token mismatch in container volume | Ensure `infra/docker-compose.yml` token matches `configs/twin_config.yaml`. |
