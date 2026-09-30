# Operational & Empirical Assumptions: Bosch Line 3 Digital Twin

All system logic, baseline thresholds, time scaling, and model architectures are grounded in explicit, documented assumptions.

---

## 1. Time Scale & Calendar Mapping
- **Assumption:** 16.75 relative time units in the Bosch dataset correspond to 1 calendar week (168 wall-clock hours).
- **Empirical Evidence:** Autocorrelation analysis on arrival timestamps across 1.18M parts revealed a prominent periodic lag at $\Delta t = 16.75$ ($r = 0.0895$).
- **Implications:**
  - $1.0\text{ unit} = 10.02985\text{ hours} \approx 601.8\text{ minutes}$.
  - $0.01\text{ unit} \approx 6.02\text{ minutes}$ (the discrete sampling resolution of the Bosch recording system).

---

## 2. Line 3 Core Manufacturing Topology
- **Assumption:** Line 3 core flow ($S29 \to S30 \to S33 \to S34 \to (S35 \mid S36) \to S37$) represents the primary production stream of the facility.
- **Empirical Evidence:**
  - 89.02% of all parts in the dataset (1,053,742 parts) visit this exact sequence.
  - S31 was decommissioned after Week 10 (handling only 3.19% of volume in the initial weeks).
  - Stations S35 and S36 operate as balanced parallel finishing branches (49.24% vs 50.76% split) with virtually identical defect distributions.

---

## 3. Station Feature Scope
- **Assumption:** Only 39 numeric features out of 968 provide meaningful operational signal; the remainder are high-missingness or zero-variance noise.
- **Empirical Evidence:**
  - 840 candidate features exhibit $>95\%$ missingness or near-zero variance.
  - Stations S34 (routing) and S37 (exit inspection) contain 0 informative numeric sensor features.
  - The 39 features are concentrated in S29 (17), S30 (10), S33 (6), S35 (2), and S36 (4).

---

## 4. Health & Risk Scoring Weights
- **Station Health Composite ($H_s \in [0, 100]$):**
  $$H_s = 100 - (0.45 \cdot S_{\text{drift}} + 0.30 \cdot S_{\text{pacing}} + 0.25 \cdot S_{\text{defect}})$$
  - Feature drift penalty ($w=0.45$): Triggers when sensor features deviate by $>3\sigma$.
  - Transit pacing penalty ($w=0.30$): Triggers when inter-station transit exceeds $1.5 \times p90$.
  - Rolling defect rate penalty ($w=0.25$): Triggers when local 100-part rolling defect rate exceeds baseline ($0.51\%$).
- **Part Risk Bands:**
  - Low Risk: $[0.0, 0.30)$
  - Medium Risk: $[0.30, 0.70)$
  - High Risk: $[0.70, 1.00]$

---

## 5. Causal Modeling Boundaries
- **Assumption:** Defect risk prediction is evaluated strictly at the entry of the S34 fork before physical entry to S35 or S36.
- **Leakage Prevention:**
  - No features from S35, S36, or S37 are included in the part-level defect feature vector.
  - `recent_defect_rate` strictly filters for parts that have completed exit inspection at S37 *prior* to the evaluated part's decision time.
