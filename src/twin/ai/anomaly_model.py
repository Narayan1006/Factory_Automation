"""
Phase C: LSTM Autoencoder for Line 3 Anomaly Detection.
Implements:
  1. Sequence Builder: Constructs sequences of L=24 consecutive non-gap 1-sim-hour windows.
     Guarantees sequences NEVER cross or span production gaps.
  2. Encoder-Decoder LSTM in PyTorch (reconstructing 35-dim station telemetry vectors).
  3. "Normal" Regime Training: Excludes hold-out scenario block and all 9 historical anomaly events.
  4. Anomaly Scoring & Attribution: Latest-window reconstruction error, threshold at p99 of normal validation,
     per-station percentage contribution breakdown.
  5. Scenario Benchmarking: Evaluates detection timing, lead time, and false positive rates.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from twin.ai.utils import PureStandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AnomalyModel")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORE_STATIONS = [29, 30, 33, 34, 35, 36, 37]
METRICS_PER_STATION = ["max_abs_z", "mean_abs_z", "p90_transit", "zero_delta_share", "throughput"]


# ------------------------------------------------------------------------------
# 1. PyTorch LSTM Autoencoder Architecture
# ------------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    """
    Encoder-Decoder LSTM for multivariate temporal reconstruction.
    Encoder maps (seq_len, in_dim) -> latent hidden state.
    Decoder reconstructs sequence from repeated latent representation.
    """

    def __init__(self, in_features: int = 35, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # Encode
        _, (h_n, _) = self.encoder(x)
        # Latent context from last layer: (batch_size, hidden_dim)
        latent = h_n[-1]

        # Repeat latent context across time steps for decoding
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        dec_out, _ = self.decoder(repeated)
        recon = self.output_layer(dec_out)
        return recon


# ------------------------------------------------------------------------------
# 2. Sequence Builder (Strictly No Gap Spanning)
# ------------------------------------------------------------------------------
def build_non_gap_sequences(
    df_windows: pd.DataFrame,
    seq_length: int = 24,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Constructs sequences of L consecutive non-gap windows.
    If window_id has a gap (is_gap=True or discontinuous window_id), the sequence resets.
    Returns:
      sequences: np.ndarray of shape (N, seq_length, n_features)
      latest_window_ids: np.ndarray of window_id for the L-th window
      metadata: List of dicts with t_start, t_end, week of latest window
    """
    if feature_cols is None:
        feature_cols = []
        for s in CORE_STATIONS:
            for m in METRICS_PER_STATION:
                feature_cols.append(f"S{s}_{m}")

    sequences = []
    latest_ids = []
    metadata = []

    # Sort strictly by window_id
    df_sorted = df_windows.sort_values("window_id").reset_index(drop=True)

    current_seq = []
    current_meta = []
    prev_w_id = None

    for _, row in df_sorted.iterrows():
        w_id = int(row["window_id"])
        is_gap = bool(row["is_gap"])

        # Check if consecutive and not a gap
        is_consecutive = (prev_w_id is None) or (w_id == prev_w_id + 1)
        prev_w_id = w_id

        if is_gap or not is_consecutive:
            # Reset sequence on gap or discontinuity
            current_seq = []
            current_meta = []
            if is_gap:
                continue

        # Add valid window
        vec = row[feature_cols].values.astype(np.float32)
        current_seq.append(vec)
        current_meta.append({
            "window_id": w_id,
            "t_start": float(row["t_start"]),
            "t_end": float(row["t_end"]),
            "week": int(row["week"]),
        })

        # When we have reached L consecutive valid windows
        if len(current_seq) == seq_length:
            sequences.append(np.array(current_seq))
            latest_ids.append(w_id)
            metadata.append(current_meta[-1]) # Info for latest (decision) window
            # Slide window by 1
            current_seq = current_seq[1:]
            current_meta = current_meta[1:]

    if sequences:
        X = np.stack(sequences, axis=0)
    else:
        X = np.empty((0, seq_length, len(feature_cols)), dtype=np.float32)

    logger.info(
        f"Built {len(X):,} non-gap sequences of length {seq_length} "
        f"(Input Dim: {len(feature_cols)} across {len(CORE_STATIONS)} stations)."
    )
    return X, np.array(latest_ids), metadata


# ------------------------------------------------------------------------------
# 3. Anomaly Mask & Training Set Filtering
# ------------------------------------------------------------------------------
def filter_normal_training_windows(
    metadata: List[Dict[str, Any]],
    anomaly_events_csv: str = "data_prep/output/grouped_anomaly_events.csv",
    exclude_scenario: Optional[str] = None,
    scenarios_config: Optional[Dict[str, Any]] = None,
    margin_units: float = 3.0,
) -> np.ndarray:
    """
    Creates boolean mask for training sequences:
    Excludes any sequence whose latest window falls into:
      1. Known historical anomaly events in grouped_anomaly_events.csv
      2. The held-out scenario window (plus margin)
    Ensures the autoencoder trains ONLY on clean, normal factory pacing.
    """
    df_anom = pd.read_csv(anomaly_events_csv)
    # Collect blocked intervals [t0 - margin, t1 + margin]
    blocked_intervals = []
    for _, r in df_anom.iterrows():
        blocked_intervals.append((float(r["t_start"]) - margin_units, float(r["t_end"]) + margin_units))

    if exclude_scenario and scenarios_config and exclude_scenario in scenarios_config:
        sc = scenarios_config[exclude_scenario]
        blocked_intervals.append((sc["t_start"] - margin_units, sc["t_end"] + margin_units))

    normal_mask = []
    for meta in metadata:
        t_w = meta["t_start"]
        in_blocked = any(t0 <= t_w <= t1 for t0, t1 in blocked_intervals)
        normal_mask.append(not in_blocked)

    normal_mask = np.array(normal_mask, dtype=bool)
    logger.info(
        f"Filtered {normal_mask.sum():,} normal sequences for training "
        f"({(~normal_mask).sum():,} blocked by anomaly/scenario hold-out)."
    )
    return normal_mask


# ------------------------------------------------------------------------------
# 4. Model Training & Threshold Fitting
# ------------------------------------------------------------------------------
def train_anomaly_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    hidden_dim: int = 32,
    num_layers: int = 1,
    lr: float = 0.002,
    batch_size: int = 64,
    max_epochs: int = 30,
    patience: int = 6,
    threshold_percentile: float = 99.0,
    random_seed: int = 42,
) -> Tuple[LSTMAutoencoder, PureStandardScaler, float, Dict[str, Any]]:
    """
    Trains LSTM Autoencoder to reconstruct normal sequences.
    Fits threshold as p99 of reconstruction error on normal validation set.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    N_train, seq_len, in_dim = X_train.shape
    N_val = X_val.shape[0]

    # Fit 2D PureStandardScaler across all time steps of training data
    scaler = PureStandardScaler()
    X_train_flat = X_train.reshape(-1, in_dim)
    scaler.fit(X_train_flat)

    # Scale data
    X_train_scaled = scaler.transform(X_train_flat).reshape(N_train, seq_len, in_dim)
    X_val_scaled = scaler.transform(X_val.reshape(-1, in_dim)).reshape(N_val, seq_len, in_dim)

    # DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train_scaled).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(in_features=in_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    X_val_t = torch.from_numpy(X_val_scaled).float().to(DEVICE)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for (b_x,) in train_loader:
            b_x = b_x.to(DEVICE)
            optimizer.zero_grad()
            recon = model(b_x)
            loss = criterion(recon, b_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_x)

        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (Best Val MSE: {best_val_loss:.5f})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Compute reconstruction errors on validation set for latest window (index -1)
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val_t).cpu().numpy()
        # Error on latest window: shape (N_val, in_dim)
        latest_err = (X_val_scaled[:, -1, :] - val_recon[:, -1, :]) ** 2
        window_errors = np.mean(latest_err, axis=1)

    threshold = float(np.percentile(window_errors, threshold_percentile))
    logger.info(f"Fitted anomaly threshold ({threshold_percentile}th percentile): {threshold:.5f}")

    training_info = {
        "best_val_loss": float(best_val_loss),
        "threshold": float(threshold),
        "threshold_percentile": float(threshold_percentile),
        "epochs_trained": epoch + 1,
    }
    return model, scaler, threshold, training_info


# ------------------------------------------------------------------------------
# 5. Score Sequence & Station Attribution
# ------------------------------------------------------------------------------
def score_window_sequence(
    model: LSTMAutoencoder,
    scaler: PureStandardScaler,
    threshold: float,
    sequence: np.ndarray,
) -> Tuple[float, Dict[str, float], bool]:

    """
    Scores a single sequence of shape (seq_len, in_dim).
    Returns:
      reconstruction_error (float)
      per_station_contributions (dict: station_id -> share in [0, 1.0])
      is_flagged (bool)
    """
    model.eval()
    seq_len, in_dim = sequence.shape
    seq_scaled = scaler.transform(sequence)
    x_t = torch.from_numpy(seq_scaled).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon = model(x_t).cpu().numpy().squeeze(0)

    # Error on the latest window
    latest_diff = (seq_scaled[-1] - recon[-1]) ** 2
    total_error = float(np.mean(latest_diff))
    is_flagged = bool(total_error >= threshold)

    # Per-station contribution breakdown
    # 5 features per station in CORE_STATIONS: [29, 30, 33, 34, 35, 36, 37]
    contributions = {}
    sum_err = max(1e-8, float(np.sum(latest_diff)))

    for i, s_num in enumerate(CORE_STATIONS):
        stn_id = f"S{s_num}"
        stn_slice = latest_diff[i * 5 : (i + 1) * 5]
        stn_err = float(np.sum(stn_slice))
        contributions[stn_id] = round(stn_err / sum_err, 4)

    return round(total_error, 5), contributions, is_flagged
