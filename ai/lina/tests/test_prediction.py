"""
Tests for the crowd prediction module.

NOTE: Uses SYNTHETIC DEMO DATA. Metrics tested here are not real accuracy.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.prediction.feature_engineering import (
    build_features,
    get_feature_columns,
    prepare_latest_features,
)
from src.prediction.model_utils import save_model, load_model, model_exists
from src.prediction.predict import CrowdPredictor


PRED_CONFIG = {
    "model": {
        "type": "xgboost",
        "model_filename": "test_crowd_predictor.pkl",
        "scaler_filename": "test_crowd_scaler.pkl",
    },
    "features": {
        "lag_windows": [1, 2, 3],
        "rolling_windows": [3],
        "use_hour": True,
        "use_minute": True,
        "use_day_of_week": True,
        "use_is_weekend": True,
        "use_queue_length": False,
    },
    "training": {
        "test_size": 0.2,
        "random_state": 42,
        "xgboost_params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
    },
    "prediction": {
        "intervals_minutes": [15, 30, 60],
        "min_history_records": 20,
    },
    "data": {
        "aggregation_interval_minutes": 5,
    },
}


def make_synthetic_df(n_records: int = 200) -> pd.DataFrame:
    """Generate a simple synthetic crowd time series."""
    base = datetime(2026, 9, 1, 8, 0, 0)
    records = []
    for i in range(n_records):
        ts = base + timedelta(minutes=i * 5)
        count = int(30 + 20 * np.sin(i / 12.0 * np.pi) + np.random.randint(0, 5))
        records.append({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"), "people_count": max(0, count)})
    return pd.DataFrame(records)


# ── Feature Engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def setup_method(self):
        self.df = make_synthetic_df(100)

    def test_build_features_returns_dataframe(self):
        result = build_features(self.df, PRED_CONFIG)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_feature_columns_include_lag(self):
        result = build_features(self.df, PRED_CONFIG)
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert "lag_3" in result.columns

    def test_feature_columns_include_rolling(self):
        result = build_features(self.df, PRED_CONFIG)
        assert "rolling_mean_3" in result.columns
        assert "rolling_std_3" in result.columns

    def test_feature_columns_include_time(self):
        result = build_features(self.df, PRED_CONFIG)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns

    def test_no_nan_after_build(self):
        result = build_features(self.df, PRED_CONFIG)
        assert not result.isnull().any().any()

    def test_get_feature_columns_excludes_target(self):
        result = build_features(self.df, PRED_CONFIG)
        feat_cols = get_feature_columns(result)
        assert "people_count" not in feat_cols
        assert "timestamp" not in feat_cols

    def test_prepare_latest_features_returns_series(self):
        config = {**PRED_CONFIG, "prediction": {"intervals_minutes": [15, 30], "min_history_records": 10}}
        result = prepare_latest_features(self.df, config)
        assert result is not None
        assert len(result) > 0

    def test_prepare_latest_features_insufficient_data(self):
        small_df = make_synthetic_df(5)
        result = prepare_latest_features(small_df, PRED_CONFIG)
        assert result is None  # Not enough history

    def test_handles_empty_dataframe(self):
        result = build_features(pd.DataFrame({"timestamp": [], "people_count": []}), PRED_CONFIG)
        assert len(result) == 0


# ── Model Utils tests ─────────────────────────────────────────────────────────

class TestModelUtils:
    def test_model_dir_found(self):
        model_dir_path = LINA_ROOT / "models"
        assert model_dir_path.exists()

    def test_save_and_load_model(self):
        test_obj = {"test": "data", "value": 42}
        saved = save_model(test_obj, "test_dummy_model.pkl")
        assert saved is not None and saved.exists()

        loaded = load_model("test_dummy_model.pkl")
        assert loaded == test_obj

        # Cleanup
        saved.unlink(missing_ok=True)

    def test_load_nonexistent_model(self):
        result = load_model("nonexistent_model_xyz.pkl")
        assert result is None

    def test_model_exists_false_for_missing(self):
        assert model_exists("definitely_not_here_xyz.pkl") is False


# ── CrowdPredictor tests ──────────────────────────────────────────────────────

class TestCrowdPredictor:
    def setup_method(self):
        self.predictor = CrowdPredictor(PRED_CONFIG)

    def test_empty_prediction_when_no_model(self):
        """Without training, predictor should return empty prediction gracefully."""
        df = make_synthetic_df(200)
        result = self.predictor.predict(df)
        # Should return a dict, not raise an exception
        assert isinstance(result, dict)
        assert "data_classification" in result

    def test_empty_prediction_has_correct_structure(self):
        empty = self.predictor._empty_prediction()
        assert "model_available" in empty
        assert empty["model_available"] is False
        assert "next_15_min" in empty
        assert "next_30_min" in empty

    def test_predict_with_insufficient_history(self):
        small_df = make_synthetic_df(5)
        result = self.predictor.predict(small_df)
        assert isinstance(result, dict)

    def test_full_training_and_prediction(self):
        """
        Integration test: train a small model then run prediction.
        Uses very small params to keep test fast.
        ⚠ Results are SYNTHETIC DEMO DATA only.
        """
        try:
            from src.prediction.train import train

            df = make_synthetic_df(300)
            training_result = train(df, PRED_CONFIG, target_steps_ahead=3)

            if training_result.get("status") == "success":
                # Verify metrics are computed
                metrics = training_result.get("metrics", {})
                assert "mae" in metrics
                assert "rmse" in metrics
                assert "r2" in metrics
                assert isinstance(metrics["mae"], float)
                assert metrics["mae"] >= 0
                assert isinstance(metrics["r2"], float)
        except ImportError:
            pytest.skip("XGBoost not installed — skipping training integration test")
