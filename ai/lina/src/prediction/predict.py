"""
Crowd Count Predictor.

Loads trained XGBoost model + scaler and generates crowd predictions
for multiple future time intervals (15 min, 30 min, 60 min).

NOTE: Predictions based on SYNTHETIC DEMO DATA are for demonstration only.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.prediction.feature_engineering import prepare_latest_features
from src.prediction.model_utils import load_model, model_exists
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrowdPredictor:
    """
    Generates crowd count predictions for future time intervals.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Loaded prediction_config.json dict.
        """
        self.config = config
        model_cfg = config.get("model", {})
        self._model = None
        self._scaler = None
        self._feature_cols = None
        self._model_filename = model_cfg.get("model_filename", "crowd_predictor.pkl")
        self._scaler_filename = model_cfg.get("scaler_filename", "crowd_scaler.pkl")

    def load(self) -> bool:
        """
        Load the model and scaler from the models directory.

        Returns:
            True if both model and scaler loaded successfully.
        """
        if not model_exists(self._model_filename):
            logger.warning("Prediction model not found: %s. Run train_prediction_model.py first.", self._model_filename)
            return False

        self._model = load_model(self._model_filename)
        self._scaler = load_model(self._scaler_filename)
        self._feature_cols = load_model("feature_columns.pkl")

        if self._model is None or self._scaler is None:
            logger.error("Failed to load model or scaler")
            return False

        logger.info("Prediction model loaded successfully")
        return True

    def predict(self, historical_df: pd.DataFrame) -> dict:
        """
        Generate predictions for configured future intervals.

        Args:
            historical_df: DataFrame with recent 'timestamp' and 'people_count' values.

        Returns:
            Dict with predictions for each configured interval.
        """
        if self._model is None:
            loaded = self.load()
            if not loaded:
                return self._empty_prediction()

        feature_row = prepare_latest_features(historical_df, self.config)
        if feature_row is None:
            logger.warning("Cannot generate predictions — insufficient history")
            return self._empty_prediction()

        intervals = self.config.get("prediction", {}).get("intervals_minutes", [15, 30, 60])
        results = {
            "data_classification": "SYNTHETIC DEMO DATA — not real accuracy",
            "model_available": True,
        }

        # Align feature row to trained feature columns
        if self._feature_cols:
            feature_row = feature_row.reindex(self._feature_cols, fill_value=0.0)

        X = feature_row.values.reshape(1, -1)
        try:
            X_scaled = self._scaler.transform(X)
        except Exception as exc:
            logger.error("Scaler transform failed: %s", exc)
            return self._empty_prediction()

        # For each interval, iteratively predict
        # (This is a simplified single-step approach — fine for demo)
        try:
            base_pred = float(self._model.predict(X_scaled)[0])
            base_pred = max(0, round(base_pred))

            for interval in intervals:
                # Simple approximation: scale by temporal factor
                scale = 1.0 + (interval / 30.0) * 0.15  # Slight growth assumption
                pred = max(0, round(base_pred * scale))
                results[f"next_{interval}_min"] = pred

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return self._empty_prediction()

        logger.info("Predictions: %s", {k: v for k, v in results.items() if k.startswith("next_")})
        return results

    def _empty_prediction(self) -> dict:
        """Return empty prediction dict when model is unavailable."""
        return {
            "data_classification": "SYNTHETIC DEMO DATA",
            "model_available": False,
            "next_15_min": None,
            "next_30_min": None,
            "next_60_min": None,
            "note": "Train the model first using scripts/train_prediction_model.py",
        }
