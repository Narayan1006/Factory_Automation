"""
Phase B: Defect-Risk Models for Bosch Line 3.
Implements:
  1. Empirical Baselines: Constant prevalence, Recent defect rate only, Logistic Regression, Shallow Non-Linear
  2. PyTorch Defect MLP (128-64, ReLU, Dropout, weighted BCE, early stopping on PR-AUC)
  3. Probability Calibration (Platt Scaling)
  4. Comprehensive Imbalanced Metrics: PR-AUC, ROC-AUC, MCC, Recall@1%, Recall@5%, Lift@1%, Brier Score
  5. Feature Ablation (with vs without recent_defect_rate)
  6. Global Permutation Feature Importance (CSV + Plot)
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from twin.ai.utils import (
    PureStandardScaler,
    PlattCalibrator,
    compute_defect_metrics,
    pure_pr_auc,
    pure_roc_auc,
    PyTorchLogisticRegression,
    PyTorchShallowClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DefectModel")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 1. PyTorch MLP Architecture
# ------------------------------------------------------------------------------
class DefectMLP(nn.Module):
    """Multi-layer Perceptron for imbalanced defect risk prediction."""

    def __init__(self, in_features: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ------------------------------------------------------------------------------
# 2. Training and Calibration Functions
# ------------------------------------------------------------------------------
def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: List[int] = [128, 64],
    dropout: float = 0.2,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 4096,
    max_epochs: int = 25,
    patience: int = 5,
    random_seed: int = 42,
) -> Tuple[DefectMLP, PlattCalibrator, Dict[str, Any]]:
    """
    Trains PyTorch DefectMLP with early stopping on validation PR-AUC,
    then calibrates probabilities via Platt Scaling.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    in_dim = X_train.shape[1]
    model = DefectMLP(in_dim, hidden_dims=hidden_dims, dropout=dropout).to(DEVICE)

    # Class imbalance weighting: pos_weight = neg / pos
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    pos_weight = torch.tensor([float(n_neg / n_pos)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Prepare PyTorch datasets
    train_ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X_train)).float(), torch.from_numpy(np.ascontiguousarray(y_train)).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_prauc = -1.0
    best_weights = None
    no_improve = 0

    X_val_t = torch.from_numpy(np.ascontiguousarray(X_val)).float().to(DEVICE)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for b_x, b_y in train_loader:
            b_x = b_x.to(DEVICE)
            b_y = b_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_y)

        # Validation PR-AUC
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
            val_prob_uncal = 1.0 / (1.0 + np.exp(-val_logits))
            val_prauc = pure_pr_auc(y_val, val_prob_uncal)

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (Best Val PR-AUC: {best_val_prauc:.5f})")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Calibrate probabilities on validation set using Platt Scaling
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy()
        val_prob_raw = 1.0 / (1.0 + np.exp(-val_logits))

    calibrator = PlattCalibrator()
    calibrator.fit(val_prob_raw, y_val)

    training_info = {
        "best_val_prauc": float(best_val_prauc),
        "epochs_trained": epoch + 1,
        "device": str(DEVICE),
    }
    return model, calibrator, training_info


# ------------------------------------------------------------------------------
# 3. Global Permutation Importance
# ------------------------------------------------------------------------------
def compute_permutation_importance(
    model: DefectMLP,
    calibrator: PlattCalibrator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    output_dir: str = "ai/output",
    n_repeats: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Computes permutation importance: drop in PR-AUC when each feature is randomly shuffled.
    Saves CSV and bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)

    # Baseline score
    X_val_t = torch.from_numpy(np.ascontiguousarray(X_val)).float().to(DEVICE)
    model.eval()
    with torch.no_grad():
        base_logits = model(X_val_t).cpu().numpy()
        base_raw = 1.0 / (1.0 + np.exp(-np.clip(base_logits, -30.0, 30.0)))
        base_prob = calibrator.predict(base_raw)
    base_prauc = pure_pr_auc(y_val, base_prob)

    importances = []
    logger.info(f"Computing permutation feature importance across {len(feature_names)} features...")

    for f_idx, fname in enumerate(feature_names):
        drop_scores = []
        for r in range(n_repeats):
            X_perm = X_val.copy()
            # Shuffle column
            np.random.shuffle(X_perm[:, f_idx])
            with torch.no_grad():
                perm_logits = model(torch.from_numpy(np.ascontiguousarray(X_perm)).float().to(DEVICE)).cpu().numpy()
                perm_raw = 1.0 / (1.0 + np.exp(-np.clip(perm_logits, -30.0, 30.0)))
                perm_prob = calibrator.predict(perm_raw)
            score_perm = pure_pr_auc(y_val, perm_prob)
            drop_scores.append(base_prauc - score_perm)

        importances.append({
            "feature": fname,
            "mean_prauc_drop": float(np.mean(drop_scores)),
            "std_prauc_drop": float(np.std(drop_scores)),
        })

    df_imp = pd.DataFrame(importances).sort_values("mean_prauc_drop", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(output_dir, "permutation_importance.csv")
    df_imp.to_csv(csv_path, index=False)

    # Plot top 20 features
    plt.figure(figsize=(10, 6))
    top20 = df_imp.head(20)
    plt.barh(range(len(top20)), top20["mean_prauc_drop"].values[::-1], color="#2b5c8f")
    plt.yticks(range(len(top20)), top20["feature"].values[::-1])
    plt.xlabel("Drop in Validation PR-AUC upon Permutation")
    plt.title("Line 3 Defect Model - Feature Importance (Permutation Drop)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "permutation_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Permutation importance saved to {csv_path} and {plot_path}")
    return df_imp
