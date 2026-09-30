"""
Pure NumPy and PyTorch Utilities for Bosch Line 3 AI Models.
Provides StandardScaler, Platt Calibrator, Imbalanced Metrics, and Baselines.
Zero external C-extension DLL dependencies for 100% stability.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 1. Pure NumPy Standard Scaler
# ------------------------------------------------------------------------------
class PureStandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PureStandardScaler":
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PureStandardScaler":
        scaler = cls()
        scaler.mean_ = np.array(d["mean"], dtype=np.float32)
        scaler.scale_ = np.array(d["scale"], dtype=np.float32)
        return scaler


# ------------------------------------------------------------------------------
# 2. Probability Calibrator (Platt Scaling)
# ------------------------------------------------------------------------------
class PlattCalibrator:
    """
    Parametric sigmoid calibration (Platt Scaling).
    Learns scalar slope A and intercept B such that P(y=1) = sigmoid(A * logit + B).
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray, epochs: int = 50, lr: float = 0.05) -> "PlattCalibrator":
        raw_probs = np.clip(np.asarray(raw_probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)
        # Convert probabilities back to unconstrained logits
        logits = np.log(raw_probs / (1.0 - raw_probs))
        y_t = torch.from_numpy(np.asarray(y_true, dtype=np.float32)).to(DEVICE)
        log_t = torch.from_numpy(logits).to(DEVICE)

        param_a = nn.Parameter(torch.tensor(1.0, device=DEVICE))
        param_b = nn.Parameter(torch.tensor(0.0, device=DEVICE))
        optimizer = optim.Adam([param_a, param_b], lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            scaled = param_a * log_t + param_b
            loss = criterion(scaled, y_t)
            loss.backward()
            optimizer.step()

        self.a = float(param_a.item())
        self.b = float(param_b.item())
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        raw_probs = np.clip(np.asarray(raw_probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)
        logits = np.log(raw_probs / (1.0 - raw_probs))
        scaled = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-scaled))

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PlattCalibrator":
        c = cls()
        c.a = float(d.get("a", 1.0))
        c.b = float(d.get("b", 0.0))
        return c


# ------------------------------------------------------------------------------
# 3. Imbalanced Metrics Suite
# ------------------------------------------------------------------------------
def pure_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under ROC via Mann-Whitney U rank-sum statistic."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    u = np.sum(ranks[y_true == 1]) - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def pure_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (Area Under PR Curve)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recalls = tp / float(n_pos)
    precisions = tp / (tp + fp)

    # Prepend (0, 1) and compute trapezoidal Riemann sum
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    delta_r = recalls[1:] - recalls[:-1]
    return float(np.sum(delta_r * precisions[1:]))


def pure_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def compute_defect_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Computes comprehensive metrics for imbalanced binary defect classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pr_auc = pure_pr_auc(y_true, y_prob)
    roc_auc = pure_roc_auc(y_true, y_prob)

    # Threshold selection via MCC grid search
    if threshold is None:
        best_mcc = -1.0
        best_th = float(np.mean(y_true))
        eval_threshs = np.unique(np.percentile(y_prob, np.linspace(50, 99.8, 80)))
        for th in eval_threshs:
            y_pred = (y_prob >= th).astype(int)
            val_mcc = pure_mcc(y_true, y_pred)
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_th = float(th)
        threshold = best_th
        mcc = best_mcc
    else:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = pure_mcc(y_true, y_pred)

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))

    if n_pos > 0:
        top_1pct = int(np.ceil(0.01 * n_total))
        top_5pct = int(np.ceil(0.05 * n_total))
        ranked_idx = np.argsort(-y_prob)

        recall_1 = float(np.sum(y_true[ranked_idx[:top_1pct]]) / n_pos)
        recall_5 = float(np.sum(y_true[ranked_idx[:top_5pct]]) / n_pos)
        lift_1 = float(recall_1 / 0.01)
    else:
        recall_1, recall_5, lift_1 = 0.0, 0.0, 0.0

    brier = float(np.mean((y_true - y_prob) ** 2))

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "mcc": round(mcc, 5),
        "optimal_threshold": round(float(threshold), 5),
        "recall_at_1pct": round(recall_1, 5),
        "recall_at_5pct": round(recall_5, 5),
        "lift_at_1pct": round(lift_1, 2),
        "brier_score": round(brier, 6),
    }


# ------------------------------------------------------------------------------
# 4. Pure PyTorch Baseline Models
# ------------------------------------------------------------------------------
class PyTorchLogisticRegression:
    """Class-weighted Logistic Regression in PyTorch."""

    def __init__(self, in_features: int, epochs: int = 20, lr: float = 0.01):
        self.in_features = in_features
        self.epochs = epochs
        self.lr = lr
        self.linear = nn.Linear(in_features, 1).to(DEVICE)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 4096) -> "PyTorchLogisticRegression":
        n_pos = max(1, int(np.sum(y == 1)))
        pos_weight = torch.tensor([float((len(y) - n_pos) / n_pos)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.linear.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.linear.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = self.linear(bx).squeeze(-1)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.linear.eval()
        with torch.no_grad():
            bx = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
            logits = self.linear(bx).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


class PyTorchShallowClassifier:
    """Non-linear 1-hidden-layer (32 units) baseline classifier."""

    def __init__(self, in_features: int, hidden_dim: int = 32, epochs: int = 20, lr: float = 0.005):
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(DEVICE)
        self.epochs = epochs
        self.lr = lr

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 4096) -> "PyTorchShallowClassifier":
        n_pos = max(1, int(np.sum(y == 1)))
        pos_weight = torch.tensor([float((len(y) - n_pos) / n_pos)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                logits = self.net(bx).squeeze(-1)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            bx = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
            logits = self.net(bx).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def df_to_markdown(df) -> str:
    """Format DataFrame as standard GitHub Markdown table without requiring tabulate."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        row_str = "| " + " | ".join([str(r[c]) for c in df.columns]) + " |"
        rows.append(row_str)
    return "\n".join([header, sep] + rows)

