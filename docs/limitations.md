# Project Limitations & Industrial Boundaries
**Project:** AI-Driven Digital Twin for Smart Factory Operations  
**HCL In-House Internship Project**

---

## 1. Dataset Anonymization Constraints
- **Lack of Physical Units:** Sensor readings are standardized float values without engineering units (temperature in °C, pressure in bar, vibration in mm/s). Consequently, anomalies are detected using statistical dispersion ($|Z|$-score) and neural reconstruction loss rather than physical process tolerance limits.
- **Anonymized Machine Types:** Stations are identified solely by numerical IDs (S29–S37). Specific equipment failure modes (e.g. bearing wear, tool chatter, nozzle clogs) cannot be physically modeled.

---

## 2. Discrete Temporal Granularity
- **Timestamp Resolution:** The Bosch dataset records timestamps in increments of 0.01 relative units (~6 minutes).
- **Transient Aggregation:** High-frequency electrical or mechanical transients occurring on millisecond cycles within a single station cycle cannot be captured; observations reflect macro-level cycle and inter-station transit dynamics.

---

## 3. Passive Digital Twin (Open-Loop Monitoring)
- **Monitoring vs Actuation:** The twin executes real-time streaming, stateful scoring, defect prediction, and alert dispatching. However, it does not issue automated closed-loop PLC feedback signals to physically halt conveyor drives or actuate pneumatic divert gates.

---

## 4. Class Imbalance Limitations
- **Base Rate Imbalance:** With an empirical defect rate of $0.51\%$ (~1 in 200 parts), standard classification accuracy is a misleading metric (a dummy model predicting all non-defects achieves 99.49% accuracy).
- **PR-AUC Upper Bound:** Due to noise, missing sensors on key routing steps, and high natural variance, model PR-AUC is bounded around $0.20–0.27$. The model must be deployed as a high-precision screening tool (prioritizing the top 1% highest-risk parts for manual QA) rather than an autonomous gatekeeper.
