# Phase 3 Live Demonstration Script: Bosch Digital Twin

This document provides a minute-by-minute live presentation guide for presenting the AI-Driven Digital Twin for Smart Factory Operations.

---

## 1. Quick Launch
To run a live scenario in one command:
```powershell
.\scripts\demo.ps1 -Scenario A -Speed 120.0
```
- Open Grafana at [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `admin_password_123`).

---

## 2. Live Scenarios Timeline

### Scenario A: Sensor Feature Drift & Co-occurring Defect Surge (Default Demo)
*Time window: $t \in [362.0, 386.0]$ (~240 hours simulated time, ~2 minutes at 120x speed)*
- **0:00 - 0:30 (Nominal Operation):**
  - **Screen:** Line Overview (`/d/line3-overview`).
  - **Talking point:** "Line 3 is operating in nominal regime. Parts traverse S29 to S37. Station health scores remain green ($>85$). Rolling defect rate matches baseline at 0.51%."
- **0:30 - 1:00 (S29 Sensor Drift Begins):**
  - **Screen:** Station Detail (`/d/line3-station-detail`), select Station `S29`.
  - **Talking point:** "Around $t=370$, notice feature `L3_S29_F3315` and `L3_S29_F3318` drifting beyond $3\sigma$. The composite station health drops to Yellow (<80) due to feature drift penalty."
- **1:00 - 1:30 (AI Defect Risk Spike at Fork):**
  - **Screen:** Parts & Risk (`/d/line3-parts-risk`) and AI Insights (`/d/line3-ai-insights`).
  - **Talking point:** "At $t \sim 372-376$, our PyTorch MLP evaluated at the S34 fork flags parts with high defect probabilities ($P > 0.40$). The AI model provides ~30-45 minutes of early lead time before parts reach exit QA at S37."
- **1:30 - 2:00 (LSTM Anomaly Flag & Resolution):**
  - **Screen:** AI Insights (`/d/line3-ai-insights`).
  - **Talking point:** "The LSTM Autoencoder reconstruction loss breaches the $p99$ threshold. Station error contribution radar highlights S29 as the primary contributor ($>45\%$). Defect burst peaks, confirmed by ground-truth labels at S37."

---

### Scenario B: Production Stoppage & Restart Defect Burst
*Time window: $t \in [492.0, 502.0]$ (~100 hours simulated time, ~50s at 120x speed)*
- **0:00 - 0:25 (Unplanned Stoppage):**
  - **Screen:** Line Overview (`/d/line3-overview`).
  - **Talking point:** "At $t=494$, output drops to zero. A ~50-hour production stoppage occurs in the historical data. Grafana triggers the 'Station Prolonged Idle' alert."
- **0:25 - 0:50 (Thermal Restart & Defect Burst):**
  - **Screen:** AI Insights (`/d/line3-ai-insights`).
  - **Talking point:** "At $t=499$, production resumes in burst mode. Immediately following restart, parts experience high defect rates. Notice the AI model immediately elevates risk on the first restarted batches, capturing the classic cold-restart phenomenon."

---

### Scenario C: Line-Wide High Density Defect Burst
*Time window: $t \in [730.0, 745.0]$ (Week 44)*
- **Highlights:** Multi-batch quality failure across parallel branches S35 and S36.
- **Focus:** Comparison between parallel branches on Station Detail (`/d/line3-station-detail`).

---

### Scenario D: Plant Shutdown (Week 51) and Cold Restart (Week 52)
*Time window: $t \in [850.0, 890.0]$*
- **Highlights:** Seasonal plant shutdown where factory output drops to zero across entire Line 3, followed by a week-long recovery trajectory.

---

## 3. Recommended Multi-Monitor Layout
- **Monitor 1 (Left):** Grafana Dashboard `Line 3 Overview` + `Line 3 AI Insights`.
- **Monitor 2 (Right):** PowerShell terminal running `.\scripts\demo.ps1` with live event telemetry logs.
