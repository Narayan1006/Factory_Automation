# Phase 3 Project Defense Sheet: 15 Core Questions & Rigorous Answers
**Project:** AI-Driven Digital Twin for Smart Factory Operations  
**HCL Internship / Technical Defense Guide**

---

### Q1: Why did you focus exclusively on Line 3 instead of modeling all four factory lines?
**Answer:**  
In our exploratory data analysis of the 1,183,747 parts in the Bosch dataset:
- **89.02% of all parts** (1,053,742 parts) visit the core sequence of Line 3 ($S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$).
- Only ~11% of parts detour through Lines 0, 1, or 2, and Station S31 was decommissioned after Week 10.
- Focusing on Line 3 captures the primary high-volume manufacturing throughput of the plant while establishing a clean, unbroken topological flow without introducing noise from low-volume custom routes.

---

### Q2: What does a time unit represent, and why is 16.75 units assumed to equal one week?
**Answer:**  
Bosch anonymized all timestamps into float values without documentation of units or epoch.
- Through autocorrelation analysis of arrival timestamps at the entry station S29, we discovered a prominent repeating seasonal lag peak at $\Delta t = 16.75$ units ($r = 0.0895$).
- Assuming standard continuous industrial weekly shift patterns (168 hours/week):
  $$\text{1.0 time unit} = \frac{168}{16.75} \approx 10.03\text{ hours (601.8 minutes)}$$
  $$\text{0.01 time unit} \approx 6.02\text{ minutes}$$
- This is an explicit, transparent working assumption documented in `configs/twin_config.yaml` rather than an unverified assertion of ground truth.

---

### Q3: What are the physical machine types for S29 through S37?
**Answer:**  
We deliberately avoid inventing machine names or fictitious physical tooling. In the Kaggle Bosch dataset, all station names, machine types, and sensor readings are strictly anonymized by Bosch to protect proprietary manufacturing intellectual property. We represent them strictly by their topological relationships, empirical cycle times, transit pacing distributions, and statistical sensor distributions.

---

### Q4: The defect model PR-AUC is ~0.20–0.27. Isn't that low? Is it useful in production?
**Answer:**  
No, in severely imbalanced manufacturing environments (base defect rate is $0.51\%$), a naive random classifier achieves a PR-AUC of only $0.0051$.
- A PR-AUC of $0.20–0.27$ represents a **40x to 50x improvement** over random baseline.
- More importantly, our model achieves a **Lift@1% of 5.8x to 7.26x**. By inspecting only the top 1% highest-risk parts flagged at the S34 fork, factory quality inspectors capture over 7% of all defect parts, saving significant manual inspection labor and preventing defective assemblies from reaching customer delivery.

---

### Q5: How did you prevent data leakage in defect prediction?
**Answer:**  
We enforced three strict causal boundaries:
1. **Decision Time Horizon:** The defect prediction is evaluated strictly at the entry of the S34 routing fork before entering S35/S36. No sensor features from downstream stations (exit S37) are included in the feature vector.
2. **Causal Defect Rate:** The rolling defect rate feature (`recent_defect_rate`) is computed strictly from parts that had already exited S37 prior to the current part's arrival at S34. Zero future exit outcomes are visible.
3. **Temporal Forward Chaining:** Folds are strictly chronological ($W[0-49] \to W[50-64] \to W[65-79] \to W[80-102]$). No future data is ever used to train earlier models.

---

### Q6: What is the architectural difference between rule-based health scoring and the AI models?
**Answer:**  
They operate as complementary defense-in-depth layers:
- **Rule-Based Scoring (`twin.core.health`):** Fast, deterministic $O(1)$ computation combining instantaneous feature $|Z|$-score drift ($w=0.45$), transit delay penalties ($w=0.30$), and rolling local defect rate ($w=0.25$). It provides immediate operator explainability.
- **PyTorch AI Models (`twin.ai`):**
  - *Part Defect MLP:* Nonlinear interaction modeling across multi-station transit bottlenecks, throughput, and informative sensors, outputting a calibrated Bayesian probability.
  - *LSTM Autoencoder:* Captures temporal sequence dynamics across 24-hour windows, flagging latent multivariate pattern anomalies before individual station thresholds trigger.

---

### Q7: Why did you train separate Leave-Scenario-Out models for each scenario?
**Answer:**  
In typical academic demos, models are evaluated on data they were trained on, resulting in optimistic bias.
- To demonstrate rigorous engineering integrity, we trained holdout models for Scenarios A, B, C, and D, where data within $[t_{\text{start}} - 3.0, t_{\text{end}} + 3.0]$ was completely excluded from training.
- During live replay of Scenario A, `twin.scoring` loads `defect_A_weights.pt` and `anomaly_A_weights.pt` via `artifacts/models/registry.json`, ensuring the running demo never tests on training data.

---

### Q8: How does the LSTM autoencoder handle production gaps and line stoppages?
**Answer:**  
In real manufacturing, factories halt for weekends, maintenance, or material shortages.
- Windows with fewer than 20 parts are explicitly marked as `is_gap = True`.
- During training and inference, consecutive sequences ($L=24$ hours) are formed strictly within contiguous non-gap operating blocks. Sequences never cross over a shutdown gap, preventing artificial reconstruction error spikes caused by zero-throughput idle periods.

---

### Q9: Why did you use Platt scaling for probability calibration?
**Answer:**  
Because manufacturing defect classes are severely imbalanced ($~200:1$), raw neural network logits or uncalibrated cross-entropy outputs suffer from probability distortion and overconfidence.
- Platt scaling fits a univariate logistic regression on held-out validation logits ($P(y=1|z) = \frac{1}{1 + e^{A z + B}}$), mapping raw scores into true posterior defect probabilities.
- This ensures that a predicted risk of $0.05$ actually corresponds to an empirical defect frequency of 5%.

---

### Q10: What operational lead time does the twin provide before parts fail at S37?
**Answer:**  
Parts reach the S34 decision fork on average **30 to 75 minutes** before completing transit through S35/S36 and exiting at S37.
- This lead time allows automated diversion to a secondary inspection spur, slowing line speed to prevent jams, or alert notification before the part is irreversibly assembled or crated.

---

### Q11: How were the 39 informative sensor features selected from the 968 original features?
**Answer:**  
Out of 968 numeric columns:
- 840 columns had missingness $>95\%$ or near-zero variance.
- Stations S34 and S37 contain 0 informative numeric features (they are purely routing and inspection checkpoints).
- We filtered features based on: (1) coverage $>5\%$, (2) non-zero standard deviation, (3) feature importance in tree-based exploration.
- This yielded exactly 39 high-signal sensor features across S29 (17), S30 (10), S33 (6), S35 (2), and S36 (4).

---

### Q12: Why do stations S35 and S36 exhibit a ~50/50 split?
**Answer:**  
Empirical data shows 49.24% of parts enter S35 and 50.76% enter S36.
- They are parallel identical operational branches designed to double throughput capacity after the S34 bottleneck.
- Defect rates across both branches are nearly identical (~0.50% vs ~0.51%), confirming balanced load distribution in nominal operation.

---

### Q13: How does the system handle cold restarts and thermal transients?
**Answer:**  
In Scenario B and Scenario D, after a prolonged stoppage ($>20$ sim-hours):
- The first 50 parts after resumption exhibit a defect rate exceeding 3.5% (7x baseline).
- The twin captures this via:
  1. Instantaneous transit delay penalties in rule-based health.
  2. The LSTM Autoencoder flagging high reconstruction error due to throughput/zero-delta shifts.
  3. The `recent_defect_rate` feedback loop immediately adjusting part risk upward.

---

### Q14: Why choose MQTT and InfluxDB v2 over traditional SQL and REST?
**Answer:**  
- **MQTT (Pub/Sub):** Factory shop floors use publish-subscribe industrial protocols (MQTT, OPC-UA) where sensors broadcast high-frequency events asynchronously without blocking callers.
- **InfluxDB v2 (TSM engine):** Optimized for time-series range aggregations, rolling window moving averages, and nano-second timestamp alignment. Querying moving averages across 100,000 sensor points executes in milliseconds via Flux vs seconds in relational SQL.
- **Decoupled Architecture:** Replay, scoring, and ingestion are independent services communicating via standard topics.

---

### Q15: What are the primary real-world limitations of this digital twin?
**Answer:**  
1. **Anonymized Sensors:** Physical engineering units (e.g. bar, Celsius, RPM) and sensor labels are withheld, requiring statistical Z-score thresholds rather than physical operational bounds.
2. **Replay vs Closed-Loop Actuation:** In this phase, the twin monitors and predicts; it does not issue automated closed-loop PLC commands to alter conveyor speeds or trigger mechanical diverters.
3. **Discrete Timestamps:** Timestamps in the dataset have a resolution of 0.01 units (~6 minutes), meaning micro-second sensor transients within a station cycle are aggregated.
