# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Train the XGBoost Crowd Prediction Model.

[WARNING] NOTE: This script trains on SYNTHETIC DEMO DATA.
Metrics printed are for demonstration purposes only.
They do NOT reflect real store prediction accuracy.

Usage:
    cd ai/lina
    python scripts/train_prediction_model.py

Output:
    Saves model files to ai/lina/models/
    Prints MAE, RMSE, R² metrics
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Make src importable from scripts/
LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.prediction.train import train
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    print("=" * 60)
    print("LINA CROWD PREDICTION -- MODEL TRAINING")
    print("[WARNING] SYNTHETIC DEMO DATA -- Not real store accuracy")
    print("=" * 60)

    # Load config
    config_path = LINA_ROOT / "config" / "prediction_config.json"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load historical data
    data_path = LINA_ROOT / "data" / "sample" / "historical_counts.csv"
    if not data_path.exists():
        print(f"ERROR: Historical data not found: {data_path}")
        print("Run this first:  python scripts/generate_sample_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} historical records")

    # Train for each prediction interval
    pred_cfg = config.get("prediction", {})
    intervals = pred_cfg.get("intervals_minutes", [15, 30, 60])
    agg_interval = config.get("data", {}).get("aggregation_interval_minutes", 5)

    for interval_min in intervals:
        steps_ahead = interval_min // agg_interval
        print(f"\n--- Training: predict +{interval_min} min ({steps_ahead} steps ahead) ---")

        # For multi-interval, we save the primary model (15-min target)
        # then note the others use scaling at inference time
        if interval_min == intervals[0]:
            result = train(df, config, target_steps_ahead=steps_ahead)
        else:
            # Informational only -- the predictor uses a scaling approach for other intervals
            print(f"  (Using {intervals[0]}-min model with runtime scaling for {interval_min}-min)")
            continue

        if result.get("status") == "success":
            metrics = result.get("metrics", {})
            print(f"\n  [OK] Model trained successfully!")
            print(f"  MAE  = {metrics.get('mae', 'N/A'):.2f} people")
            print(f"  RMSE = {metrics.get('rmse', 'N/A'):.2f} people")
            print(f"  R²   = {metrics.get('r2', 'N/A'):.4f}")
            print(f"  Training samples: {result.get('training_samples')}")
            print(f"  Test samples:     {result.get('test_samples')}")
            print(f"\n  Model saved to: {result.get('model_path')}")
        else:
            print(f"  [FAIL] Training failed: {result.get('error', 'unknown error')}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Run the demo: python scripts/run_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()


