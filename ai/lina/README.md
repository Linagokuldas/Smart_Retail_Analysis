# Lina — Crowd, Queue & Predictive Analytics Module

**Smart Retail Analytics System | Smart India Hackathon (SIH) 2026**
**Developer: Lina**
**Scope: `ai/lina/` only**

---

> ⚠ **All sample data and demo outputs in this module are SYNTHETIC DEMO DATA.**
> They are generated for development, testing, and demonstration purposes only.
> They do NOT represent real store analytics or real customer behaviour.

---

## Purpose

This module provides complete **Crowd, Queue, and Predictive Analytics** for a smart retail store. It:

- Receives person tracking events from the upstream computer vision pipeline (Kanimozhi)
- Computes zone-wise crowd counts, density, and crowd levels
- Generates customer density heatmaps
- Detects queues and estimates waiting times per billing counter
- Analyses peak hours and historical trends
- Predicts short-term future crowd using an XGBoost regression model
- Generates actionable operational recommendations
- Produces a unified JSON output ready for Deepika's backend

---

## Architecture

```
Upstream CV Pipeline (Kanimozhi)
         ↓  [Standard Tracking Events]
   InputAdapter (validate + normalize)
         ↓
   ZoneCounter (polygon PIP assignment)
         ↓
   DensityCalculator → CrowdLevelClassifier
         ↓
   HeatmapGenerator (accumulate positions)
         ↓
   QueueDetector (ROI-based)
         ↓
   WaitingTimeTracker (with grace period)
         ↓
   CounterAnalytics (NORMAL / BUSY / OVERLOADED)
         ↓
   HistoricalAnalytics + PeakHoursAnalyzer
         ↓
   CrowdPredictor (XGBoost)
         ↓
   RecommendationEngine (rule-based)
         ↓
   OutputAdapter → Unified JSON
         ↓
   Deepika's Backend
```

---

## Folder Structure

```
ai/lina/
│
├── README.md
├── INTEGRATION_CONTRACT.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── zones.json              # Store zones, queue regions, billing counters
│   ├── thresholds.json         # Crowd, queue, counter, recommendation thresholds
│   └── prediction_config.json  # XGBoost params, feature config, intervals
│
├── data/
│   ├── sample/
│   │   ├── people_tracks.json      # Synthetic tracking events
│   │   ├── queue_tracks.json       # Synthetic queue events
│   │   └── historical_counts.csv   # 30-day synthetic crowd history
│   └── generated/                  # Runtime-generated data
│
├── src/
│   ├── crowd/
│   │   ├── zone_counter.py     # Polygon-based zone assignment
│   │   ├── density.py          # Crowd density (people/sqm)
│   │   └── crowd_level.py      # LOW/MEDIUM/HIGH/CRITICAL classification
│   ├── heatmap/
│   │   └── heatmap_generator.py
│   ├── queue/
│   │   ├── queue_detector.py   # ROI-based queue detection
│   │   ├── waiting_time.py     # Per-track wait time with grace period
│   │   └── counter_analytics.py
│   ├── analytics/
│   │   ├── peak_hours.py       # Hourly/daily peak analysis
│   │   ├── historical.py       # Daily/weekly/monthly aggregation
│   │   └── recommendations.py  # Rule-based recommendations
│   ├── prediction/
│   │   ├── feature_engineering.py
│   │   ├── train.py            # XGBoost training + metrics
│   │   ├── predict.py          # Inference for 15/30/60-min intervals
│   │   └── model_utils.py      # Save/load model from ai/lina/models/
│   ├── pipeline/
│   │   └── lina_pipeline.py    # End-to-end orchestrator
│   ├── adapters/
│   │   ├── input_adapter.py    # Validates + normalizes upstream events
│   │   └── output_adapter.py   # Assembles unified backend-ready JSON
│   └── utils/
│       ├── logger.py
│       └── validators.py
│
├── models/                 # Trained model .pkl files (generated at runtime)
├── outputs/
│   ├── json/               # lina_output.json (generated at runtime)
│   ├── heatmaps/           # Heatmap PNG images (generated at runtime)
│   └── reports/            # Log files
│
├── scripts/
│   ├── generate_sample_data.py     # Generate synthetic demo data
│   ├── train_prediction_model.py   # Train XGBoost model
│   └── run_demo.py                 # Full pipeline demo
│
└── tests/
    ├── test_crowd.py
    ├── test_heatmap.py
    ├── test_queue.py
    ├── test_peak_hours.py
    ├── test_prediction.py
    └── test_pipeline.py
```

---

## Installation

```bash
# Navigate to this module
cd ai/lina

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- numpy, pandas, scipy
- scikit-learn
- xgboost
- opencv-python
- PyYAML
- pytest

---

## How to Generate Sample Data

```bash
cd ai/lina
python scripts/generate_sample_data.py
```

Creates inside `data/sample/`:
- `people_tracks.json` — 120 frames of synthetic tracking events
- `queue_tracks.json` — 60 frames of queue-specific tracking events
- `historical_counts.csv` — 30 days of 5-minute aggregated crowd counts

> ⚠ All data is clearly labeled as **SYNTHETIC DEMO DATA**.

---

## How to Train the Prediction Model

```bash
cd ai/lina
python scripts/train_prediction_model.py
```

- Loads `data/sample/historical_counts.csv`
- Trains an XGBoost regression model (target: +15 min crowd)
- Prints **MAE**, **RMSE**, and **R²** metrics
- Saves model to `models/crowd_predictor.pkl`

> ⚠ Metrics shown are for **SYNTHETIC DEMO DATA** only.

---

## How to Run the Demo

```bash
cd ai/lina
python scripts/run_demo.py
```

The demo:
1. Loads configuration from `config/`
2. Generates sample data if not already present
3. Runs the complete analytics pipeline
4. Prints a full summary to console
5. Saves `outputs/json/lina_output.json`
6. Saves `outputs/heatmaps/heatmap_*.png`

No real camera, hardware, or external services required.

---

## How to Run Prediction Standalone

```python
import json
import pandas as pd
from src.prediction.predict import CrowdPredictor

with open("config/prediction_config.json") as f:
    config = json.load(f)

predictor = CrowdPredictor(config)
predictor.load()  # Loads model from models/

df = pd.read_csv("data/sample/historical_counts.csv")
result = predictor.predict(df)
print(result)
# → {'next_15_min': 52, 'next_30_min': 59, 'next_60_min': 67, ...}
```

---

## How to Run Tests

```bash
cd ai/lina
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run a specific test file:
```bash
pytest tests/test_crowd.py -v
pytest tests/test_queue.py -v
pytest tests/test_pipeline.py -v
```

All tests run **without** external hardware, cameras, or databases.

---

## Input JSON Format

Each upstream tracking event **must** have this schema:

```json
{
    "timestamp": "2026-09-13T12:30:15",
    "camera_id": "CAM01",
    "track_id": 27,
    "class": "person",
    "confidence": 0.94,
    "bbox": [x1, y1, x2, y2],
    "center": [cx, cy]
}
```

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | ISO 8601 datetime |
| `camera_id` | string | Camera identifier |
| `track_id` | int | Unique person track ID (per session) |
| `class` | string | Must be `"person"` |
| `confidence` | float | Detection confidence in [0.0, 1.0] |
| `bbox` | list[4 floats] | `[x1, y1, x2, y2]` pixel bounding box |
| `center` | list[2 floats] | `[cx, cy]` centroid pixel coordinate |

**Invalid events are logged and skipped — they do not crash the pipeline.**

---

## Output JSON Format

```json
{
    "schema_version": "1.0.0",
    "generated_at": "2026-09-13T12:30:15.123456",
    "frame_timestamp": "2026-09-13T12:30:15",
    "data_classification": "SYNTHETIC DEMO DATA",
    "people": {
        "total": 42
    },
    "zones": {
        "ZONE_A": {
            "name": "Grocery",
            "count": 12,
            "density_ppsm": 0.1,
            "normalized_density": 0.05,
            "crowd_level": "LOW"
        }
    },
    "queues": {
        "COUNTER_1": {
            "queue_length": 8,
            "average_waiting_time_seconds": 90.0,
            "max_waiting_time_seconds": 150.0,
            "utilization": 0.8,
            "status": "BUSY",
            "queue_detected": true
        }
    },
    "peak_hours": {
        "peak_hour": "12:00-13:00",
        "busiest_day": "Saturday",
        "current_status": "HIGH",
        "current_people": 42
    },
    "prediction": {
        "data_classification": "SYNTHETIC DEMO DATA",
        "model_available": true,
        "next_15_min": 51,
        "next_30_min": 63,
        "next_60_min": 72
    },
    "recommendations": [
        {
            "type": "QUEUE_CONGESTION",
            "severity": "HIGH",
            "message": "COUNTER_1 queue exceeds threshold (length: 8).",
            "recommendation": "Consider opening another billing counter.",
            "counter_id": "COUNTER_1",
            "timestamp": "2026-09-13T12:30:15"
        }
    ]
}
```

---

## Configuration

### `config/zones.json`
Define store zones and queue regions as polygons. No zones are hard-coded in Python.

### `config/thresholds.json`
All crowd, queue, waiting-time, counter, and recommendation trigger values.
**Edit this file to tune the system — do not modify Python source.**

### `config/prediction_config.json`
XGBoost hyperparameters, feature flags, prediction intervals, and data aggregation settings.

---

## Integration Contract for Deepika (Backend)

See **`INTEGRATION_CONTRACT.md`** for the full input/output contract.

**Summary:**
- Deepika's backend should call `LinaPipeline.process_events(raw_events)` OR read `outputs/json/lina_output.json`
- The JSON schema is versioned (`schema_version: "1.0.0"`)
- Lina does **not** implement any HTTP API — that is Deepika's responsibility

---

## Limitations

- Queue detection relies on configured polygon ROIs, not a separate AI model
- Crowd prediction is trained on synthetic data — real store validation required
- Heatmap requires OpenCV; if unavailable, numerical heatmap still works
- No facial recognition or identity tracking — anonymous track IDs only
- Designed for single-store, single-session use at this prototype stage

---

## Future Improvements

- [ ] Multi-camera coordinate fusion before zone assignment
- [ ] LSTM or Transformer-based crowd prediction for better temporal modelling
- [ ] Automated zone polygon generation from store floor plan
- [ ] Real-time WebSocket output adapter for live dashboard streaming
- [ ] Edge deployment optimisation (TensorRT, ONNX export for XGBoost)
- [ ] Confidence-weighted heatmap accumulation
- [ ] Anomaly detection for unusual crowd behaviour

---

## Privacy

- No facial recognition
- No face images stored
- No customer identities
- Only anonymous integer track IDs are used
- Compliant with privacy-by-design principles
