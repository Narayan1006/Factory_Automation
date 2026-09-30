# Trained Model Artifacts

This directory contains trained PyTorch neural networks, feature scalers, probability calibrators, and the scenario model registry for the Bosch Digital Twin.

## Contents
- **Defect Risk Models** (`defect_{variant}_weights.pt`, `defect_{variant}_scaler.json`, `defect_{variant}_calibrator.json`, `defect_{variant}_metadata.json`):
  - 2-layer MLP classifier ($128 \to 64$) predicting part-level defect probability at the S34 routing fork before entry to S35/S36.
  - Calibrated with Platt scaling (logistic regression on validation logits).
  - Evaluated using forward chaining (Weeks 0-49, 50-64, 65-79, 80-102) yielding Lift@1% up to 7.26x.
- **Anomaly Detection Models** (`anomaly_{variant}_weights.pt`, `anomaly_{variant}_scaler.json`, `anomaly_{variant}_metadata.json`):
  - 1-layer LSTM Autoencoder (hidden dim 32) trained on consecutive non-gap 1-sim-hour temporal windows (sequence length $L=24$, 35 metrics across stations S29-S37).
  - Threshold calibrated at 99th percentile of normal operational reconstruction errors.
- **Model Registry** (`registry.json`):
  - Maps active live scenario (`A`, `B`, `C`, `D`) to its dedicated **Leave-Scenario-Out** holdout model, ensuring zero evaluation on data the model was trained on.
  - Maps general evaluation to `defect_general` and `anomaly_general`.

## Retraining
To retrain and re-export all model variants:
```powershell
python -m twin.ai.train_all
```
