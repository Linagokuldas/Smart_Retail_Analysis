"""
XGBoost Crowd Prediction Model Training.

IMPORTANT: This module uses SYNTHETIC DEMO DATA.
Metrics printed here reflect synthetic data performance only.
Do NOT interpret as real store prediction accuracy.

Trains an XGBoost regression model to predict:
- Next 15-minute crowd count
- Next 30-minute crowd count
- Next 60-minute crowd count (optional)

Saves trained model and scaler to ai/lina/models/
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.prediction.feature_engineering import build_features, get_feature_columns
from src.prediction.model_utils import save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train(df: pd.DataFrame, config: dict, target_steps_ahead: int = 3) -> dict:
    """
    Train an XGBoost regression model to predict crowd count N steps ahead.

    Args:
        df: Historical crowd DataFrame with 'timestamp' and 'people_count' columns.
        config: Loaded prediction_config.json dict.
        target_steps_ahead: Number of aggregation intervals ahead to predict.
                            Default=3 → 15 min at 5-min intervals.

    Returns:
        Dict containing model performance metrics and file paths.
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost is not installed. Run: pip install xgboost")
        return {"error": "XGBoost not available"}

    logger.info("=" * 60)
    logger.info("TRAINING NOTE: Using SYNTHETIC DEMO DATA")
    logger.info("Metrics below reflect synthetic data only.")
    logger.info("=" * 60)

    # Build features
    feature_df = build_features(df, config)
    if len(feature_df) < 50:
        logger.error("Insufficient data for training: only %d rows after feature engineering", len(feature_df))
        return {"error": "Insufficient training data"}

    # Create target: people_count shifted N steps ahead
    feature_df["target"] = feature_df["people_count"].shift(-target_steps_ahead)
    feature_df = feature_df.dropna(subset=["target"]).reset_index(drop=True)

    feature_cols = get_feature_columns(feature_df)
    X = feature_df[feature_cols].values
    y = feature_df["target"].values

    # Train / test split
    train_cfg = config.get("training", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost model
    xgb_params = train_cfg.get("xgboost_params", {})
    model = XGBRegressor(
        n_estimators=int(xgb_params.get("n_estimators", 200)),
        max_depth=int(xgb_params.get("max_depth", 6)),
        learning_rate=float(xgb_params.get("learning_rate", 0.05)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        reg_alpha=float(xgb_params.get("reg_alpha", 0.1)),
        reg_lambda=float(xgb_params.get("reg_lambda", 1.0)),
        random_state=random_state,
        verbosity=0,
    )

    logger.info("Training XGBoost model (target: %d steps ahead, %d training samples)...", target_steps_ahead, len(X_train))
    model.fit(X_train_scaled, y_train)

    # Metrics
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)  # Crowd count cannot be negative

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info("=" * 60)
    logger.info("MODEL METRICS (SYNTHETIC DEMO DATA — not real accuracy)")
    logger.info("  MAE  = %.2f people", mae)
    logger.info("  RMSE = %.2f people", rmse)
    logger.info("  R²   = %.4f", r2)
    logger.info("  Test samples = %d", len(y_test))
    logger.info("=" * 60)

    # Save model and scaler
    model_cfg = config.get("model", {})
    model_filename = model_cfg.get("model_filename", "crowd_predictor.pkl")
    scaler_filename = model_cfg.get("scaler_filename", "crowd_scaler.pkl")
    feature_filename = "feature_columns.pkl"

    model_path = save_model(model, model_filename)
    scaler_path = save_model(scaler, scaler_filename)
    features_path = save_model(feature_cols, feature_filename)

    return {
        "status": "success",
        "data_classification": "SYNTHETIC DEMO DATA",
        "target_steps_ahead": target_steps_ahead,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        },
        "model_path": str(model_path) if model_path else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "feature_columns": feature_cols,
    }
