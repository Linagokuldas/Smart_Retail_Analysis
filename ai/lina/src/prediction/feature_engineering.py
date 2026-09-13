"""
Feature Engineering for Crowd Prediction.

Converts raw time-series crowd count data into ML-ready features.

Features produced:
- Time features: hour, minute, day_of_week, is_weekend
- Lag features: count at t-1, t-2, t-3, t-6
- Rolling window features: rolling mean, rolling std for 3, 6, 12 steps
- Optional: recent queue length

NOTE: This uses SYNTHETIC DEMO DATA for development and testing.
Do not interpret model performance as real store prediction accuracy.
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Engineer features from a time-series crowd DataFrame.

    Args:
        df: DataFrame with at least 'timestamp' and 'people_count' columns.
            Optional: 'queue_length' column.
        config: Loaded prediction_config.json dict.

    Returns:
        Feature DataFrame with NaN rows dropped (caused by lag/rolling windows).
    """
    feature_cfg = config.get("features", {})
    lag_windows = feature_cfg.get("lag_windows", [1, 2, 3, 6])
    rolling_windows = feature_cfg.get("rolling_windows", [3, 6, 12])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    if feature_cfg.get("use_hour", True):
        df["hour"] = df["timestamp"].dt.hour

    if feature_cfg.get("use_minute", True):
        df["minute"] = df["timestamp"].dt.minute

    if feature_cfg.get("use_day_of_week", True):
        df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun

    if feature_cfg.get("use_is_weekend", True):
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    # Lag features
    for lag in lag_windows:
        df[f"lag_{lag}"] = df["people_count"].shift(lag)

    # Rolling window features
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = (
            df["people_count"].shift(1).rolling(window=window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["people_count"].shift(1).rolling(window=window).std()
        )

    # Optional queue length feature
    if feature_cfg.get("use_queue_length", True) and "queue_length" in df.columns:
        df["recent_queue_length"] = df["queue_length"].shift(1)

    # Drop NaN rows created by lag/rolling operations
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("Feature engineering: dropped %d rows with NaN (expected from lag/rolling)", dropped)

    logger.info("Feature matrix built: %d rows × %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (excluding target, metadata, and
    any non-numeric columns that cannot be used in the ML model).

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List of numeric feature column names suitable for model training.
    """
    # Explicit metadata columns to exclude
    non_features = {"timestamp", "people_count", "queue_length", "zone_id"}
    # Also exclude any column whose name starts with underscore (e.g. _synthetic, _queue_region)
    # and any column that is not numeric dtype
    result = []
    for c in df.columns:
        if c in non_features:
            continue
        if c.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            result.append(c)
    return result


def prepare_latest_features(
    df: pd.DataFrame,
    config: dict,
    expected_columns: Optional[list] = None,
) -> Optional[pd.Series]:
    """
    Build a single feature row from the latest available data
    (for real-time prediction).

    Args:
        df: Historical DataFrame (recent records).
        config: Loaded prediction_config.json.
        expected_columns: If provided, the output Series will be reindexed to
            exactly match these column names (filling any missing with 0.0).
            Pass the feature_columns list saved at training time.

    Returns:
        Feature Series for the next time step, or None if insufficient data.
    """
    feature_cfg = config.get("features", {})
    lag_windows = feature_cfg.get("lag_windows", [1, 2, 3, 6])
    rolling_windows = feature_cfg.get("rolling_windows", [3, 6, 12])
    min_history = config.get("prediction", {}).get("min_history_records", 50)

    if len(df) < min_history:
        logger.warning(
            "Insufficient history for prediction: have %d, need %d records", len(df), min_history
        )
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    last = df.iloc[-1]
    next_ts = last["timestamp"] + pd.Timedelta(
        minutes=config.get("data", {}).get("aggregation_interval_minutes", 5)
    )

    row = {}

    if feature_cfg.get("use_hour", True):
        row["hour"] = next_ts.hour
    if feature_cfg.get("use_minute", True):
        row["minute"] = next_ts.minute
    if feature_cfg.get("use_day_of_week", True):
        row["day_of_week"] = next_ts.dayofweek
    if feature_cfg.get("use_is_weekend", True):
        row["is_weekend"] = int(next_ts.dayofweek >= 5)

    counts = df["people_count"].values
    for lag in lag_windows:
        row[f"lag_{lag}"] = float(counts[-lag]) if len(counts) >= lag else 0.0

    for window in rolling_windows:
        if len(counts) >= window:
            row[f"rolling_mean_{window}"] = float(np.mean(counts[-window:]))
            row[f"rolling_std_{window}"] = float(np.std(counts[-window:]))
        else:
            row[f"rolling_mean_{window}"] = float(np.mean(counts))
            row[f"rolling_std_{window}"] = float(np.std(counts)) if len(counts) > 1 else 0.0

    if feature_cfg.get("use_queue_length", True) and "queue_length" in df.columns:
        row["recent_queue_length"] = float(df["queue_length"].iloc[-1])

    series = pd.Series(row)

    # Align to training feature columns if provided
    if expected_columns:
        series = series.reindex(expected_columns, fill_value=0.0)

    return series
