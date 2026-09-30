"""
Phase D: Stable Real-Time Inference Loader.
Provides:
  - predict_part(features_dict) -> calibrated defect probability
  - score_window(sequence) -> (reconstruction_error, per_station_contributions, is_flagged)
Handles median imputation, missing indicators, scaling, and calibration seamlessly.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from twin.ai.defect_model import DefectMLP
from twin.ai.anomaly_model import LSTMAutoencoder, score_window_sequence
from twin.ai.utils import PureStandardScaler, PlattCalibrator
from twin.paths import MODELS_DIR

logger = logging.getLogger("AIInference")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AIInferenceService:
    """Manages loaded defect and anomaly models for real-time scoring."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        scenario: Optional[str] = None,
    ):
        self.models_dir = str(models_dir) if models_dir is not None else str(MODELS_DIR)
        self.scenario = scenario
        self.defect_model: Optional[DefectMLP] = None
        self.defect_scaler: Optional[PureStandardScaler] = None
        self.defect_calibrator: Optional[PlattCalibrator] = None
        self.defect_meta: Dict[str, Any] = {}
        self.anomaly_model: Optional[LSTMAutoencoder] = None
        self.anomaly_scaler: Optional[PureStandardScaler] = None
        self.anomaly_meta: Dict[str, Any] = {}
        self.anomaly_threshold = 0.0

        self.load_models(scenario)

    def load_models(self, scenario: Optional[str] = None):
        """Loads models according to scenario registry or general production default."""
        self.scenario = scenario
        reg_file = os.path.join(self.models_dir, "registry.json")

        variant = "general"
        if scenario and os.path.exists(reg_file):
            try:
                with open(reg_file, "r") as f:
                    reg = json.load(f)
                if scenario in reg.get("scenarios", {}):
                    variant = scenario
            except Exception as e:
                logger.warning(f"Could not parse registry.json: {e}")

        logger.info(f"Loading AI Model variant: '{variant}' (Scenario: {scenario or 'General'})")

        # 1. Load Defect Model
        def_meta_file = os.path.join(self.models_dir, f"defect_{variant}_metadata.json")
        if not os.path.exists(def_meta_file):
            def_meta_file = os.path.join(self.models_dir, "defect_general_metadata.json")

        if os.path.exists(def_meta_file):
            with open(def_meta_file, "r") as f:
                self.defect_meta = json.load(f)

            weights_path = os.path.join(self.models_dir, self.defect_meta["weights_file"])
            scaler_path = os.path.join(self.models_dir, self.defect_meta["scaler_file"])
            calib_path = os.path.join(self.models_dir, self.defect_meta["calibrator_file"])

            in_features = self.defect_meta["in_features"]
            self.defect_model = DefectMLP(in_features=in_features).to(DEVICE)
            self.defect_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            self.defect_model.eval()

            with open(scaler_path, "r", encoding="utf-8") as f:
                self.defect_scaler = PureStandardScaler.from_dict(json.load(f))
            with open(calib_path, "r", encoding="utf-8") as f:
                self.defect_calibrator = PlattCalibrator.from_dict(json.load(f))

            logger.info(f"Loaded Defect Model: {self.defect_meta['variant']} ({in_features} features)")
        else:
            logger.warning(f"Defect model metadata {def_meta_file} not found.")

        # 2. Load Anomaly Model
        anom_meta_file = os.path.join(self.models_dir, f"anomaly_{variant}_metadata.json")
        if not os.path.exists(anom_meta_file):
            anom_meta_file = os.path.join(self.models_dir, "anomaly_general_metadata.json")

        if os.path.exists(anom_meta_file):
            with open(anom_meta_file, "r") as f:
                self.anomaly_meta = json.load(f)

            weights_path = os.path.join(self.models_dir, self.anomaly_meta["weights_file"])
            scaler_path = os.path.join(self.models_dir, self.anomaly_meta["scaler_file"])

            in_features = self.anomaly_meta["in_features"]
            self.anomaly_threshold = float(self.anomaly_meta["threshold"])

            self.anomaly_model = LSTMAutoencoder(in_features=in_features).to(DEVICE)
            self.anomaly_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            self.anomaly_model.eval()

            with open(scaler_path, "r", encoding="utf-8") as f:
                self.anomaly_scaler = PureStandardScaler.from_dict(json.load(f))

            logger.info(f"Loaded Anomaly Model: {self.anomaly_meta['variant']} (Threshold: {self.anomaly_threshold:.4f})")
        else:
            logger.warning(f"Anomaly model metadata {anom_meta_file} not found.")

    def predict_part(self, features_dict: Dict[str, Any]) -> float:
        """
        Predict calibrated defect probability for a single part at S35/S36 decision time.
        Takes dict of raw features and metadata.
        """
        if self.defect_model is None or self.defect_scaler is None:
            return 0.005069

        feature_names = self.defect_meta["feature_names"]
        train_medians = self.defect_meta.get("train_medians", {})

        vec = []
        for col in feature_names:
            if col.endswith("_isna"):
                raw_col = col[:-5]
                val = 1.0 if raw_col not in features_dict or features_dict[raw_col] is None or np.isnan(features_dict[raw_col]) else 0.0
            else:
                raw_val = features_dict.get(col)
                if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                    val = float(train_medians.get(col, 0.0))
                else:
                    val = float(raw_val)
            vec.append(val)

        X = np.array([vec], dtype=np.float32)
        X_scaled = self.defect_scaler.transform(X)

        with torch.no_grad():
            x_t = torch.from_numpy(X_scaled).float().to(DEVICE)
            logit = self.defect_model(x_t).cpu().numpy()[0]
            p_raw = 1.0 / (1.0 + np.exp(-logit))

        if self.defect_calibrator is not None:
            p_cal = float(self.defect_calibrator.predict(np.array([p_raw]))[0])
        else:
            p_cal = float(p_raw)

        return round(float(p_cal), 5)

    def score_window(self, sequence: np.ndarray) -> Tuple[float, Dict[str, float], bool]:
        """
        Score a sequence of shape (seq_len, 35).
        Returns (reconstruction_score, per_station_contributions, is_flagged).
        """
        if self.anomaly_model is None or self.anomaly_scaler is None:
            return 0.0, {}, False

        return score_window_sequence(
            self.anomaly_model,
            self.anomaly_scaler,
            self.anomaly_threshold,
            sequence,
        )
