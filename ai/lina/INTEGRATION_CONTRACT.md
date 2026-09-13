# Lina Module — Integration Contract

**Module:** Crowd, Queue & Predictive Analytics
**Developer:** Lina
**Version:** 1.0.0
**Last Updated:** 2026-09-13

---

## Overview

This document defines the **exact interface** between Lina's analytics module
and the surrounding system. Other team members should read this document — not
Lina's source code — to understand how to integrate.

---

## 1. What Lina Expects from Upstream (Kanimozhi)

### Input Format

Lina's module accepts a **list of JSON objects**, one per person detection event.

Each event **must** contain:

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

#### Field Specification

| Field | Type | Required | Notes |
|---|---|---|---|
| `timestamp` | string | ✅ | ISO 8601 (`YYYY-MM-DDTHH:MM:SS`) |
| `camera_id` | string | ✅ | Any string identifier |
| `track_id` | int | ✅ | Unique per tracked person; persists across frames |
| `class` | string | ✅ | Must be `"person"` for analytics to apply |
| `confidence` | float | ✅ | In range [0.0, 1.0] |
| `bbox` | list[float, 4] | ✅ | `[x1, y1, x2, y2]` in pixel coordinates |
| `center` | list[float, 2] | ✅ | `[cx, cy]` centroid in pixel coordinates |

#### Validation Rules (enforced by Lina)
- Events missing any required field → **skipped with warning log** (pipeline continues)
- `confidence < 0.50` → filtered out by default (configurable)
- Non-numeric `bbox` or `center` → skipped
- `bbox` with zero or negative dimensions → skipped
- Empty `timestamp` string → skipped

#### Coordinate System
- Origin `(0, 0)` = **top-left corner** of the camera frame
- Default frame dimensions: **1000 × 800 pixels** (configurable in pipeline init)
- Coordinates should match `config/zones.json` polygon definitions

### How Kanimozhi Delivers Input

**Option A — Python function call (preferred for co-location):**
```python
from src.pipeline.lina_pipeline import LinaPipeline

pipeline = LinaPipeline(zones_config, thresholds, pred_config)
output = pipeline.process_events(raw_events_list)
```

**Option B — JSON file (batch/async mode):**
```python
output = pipeline.process_json_file("path/to/tracking_events.json")
```

**Option C — Real-time stream (future):**
Call `pipeline.process_events([event])` per frame in a loop.

---

## 2. What Lina Produces for Deepika's Backend

### Output Format

Lina produces a **single unified JSON object** per analytics cycle.

```json
{
    "schema_version": "1.0.0",
    "generated_at": "2026-09-13T12:30:15.123456",
    "frame_timestamp": "2026-09-13T12:30:15",
    "data_classification": "SYNTHETIC DEMO DATA — Not real store analytics",

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
        },
        "ZONE_B": {
            "name": "Beverages",
            "count": 7,
            "density_ppsm": 0.058,
            "normalized_density": 0.029,
            "crowd_level": "LOW"
        },
        "BILLING": {
            "name": "Billing Area",
            "count": 15,
            "density_ppsm": 0.1875,
            "normalized_density": 0.09,
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
        },
        "COUNTER_2": {
            "queue_length": 3,
            "average_waiting_time_seconds": 30.0,
            "max_waiting_time_seconds": 60.0,
            "utilization": 0.3,
            "status": "NORMAL",
            "queue_detected": true
        },
        "COUNTER_3": {
            "queue_length": 0,
            "average_waiting_time_seconds": 0.0,
            "max_waiting_time_seconds": 0.0,
            "utilization": 0.0,
            "status": "NORMAL",
            "queue_detected": false
        }
    },

    "peak_hours": {
        "peak_hour": "12:00-13:00",
        "peak_hour_avg_people": 75.5,
        "min_traffic_hour": "08:00-09:00",
        "busiest_day": "Saturday",
        "overall_avg_people": 48.2,
        "overall_max_people": 95,
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
    ],

    "heatmap_path": "outputs/heatmaps/heatmap_20260913_123015.png"
}
```

### Output Field Reference

#### `people`
| Field | Type | Description |
|---|---|---|
| `total` | int | Total unique persons in latest frame |

#### `zones[zone_id]`
| Field | Type | Description |
|---|---|---|
| `name` | string | Human-readable zone name |
| `count` | int | People count in this zone |
| `density_ppsm` | float | People per square metre |
| `normalized_density` | float | Density normalised to [0, 1] |
| `crowd_level` | string | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |

#### `queues[counter_id]`
| Field | Type | Description |
|---|---|---|
| `queue_length` | int | Current people in queue |
| `average_waiting_time_seconds` | float | Mean wait time for completed waits |
| `max_waiting_time_seconds` | float | Maximum recorded wait time |
| `utilization` | float | Queue fill ratio [0, 1] |
| `status` | string | `NORMAL` / `BUSY` / `OVERLOADED` |
| `queue_detected` | bool | True if queue threshold met |

#### `peak_hours`
| Field | Type | Description |
|---|---|---|
| `peak_hour` | string | Busiest hour label e.g. `"12:00-13:00"` |
| `busiest_day` | string | Day of week with highest average traffic |
| `current_status` | string | `LOW` / `MEDIUM` / `HIGH` |
| `current_people` | int | People count in current frame |

#### `prediction`
| Field | Type | Description |
|---|---|---|
| `model_available` | bool | False if model not yet trained |
| `next_15_min` | int \| null | Predicted crowd in 15 min |
| `next_30_min` | int \| null | Predicted crowd in 30 min |
| `next_60_min` | int \| null | Predicted crowd in 60 min |

#### `recommendations[]`
| Field | Type | Description |
|---|---|---|
| `type` | string | e.g. `QUEUE_CONGESTION`, `ZONE_HIGH_DENSITY` |
| `severity` | string | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `message` | string | Human-readable description |
| `recommendation` | string | Suggested action |
| `timestamp` | string | When generated |
| `zone_id` | string | (optional) Related zone |
| `counter_id` | string | (optional) Related counter |

---

## 3. How Deepika's Backend Should Consume the Output

### Option A — Direct Python call (monolith / same process)
```python
from src.pipeline.lina_pipeline import LinaPipeline

# Initialise once at startup
pipeline = LinaPipeline(zones_config, thresholds, pred_config)

# Call per frame / per analytics cycle
output = pipeline.process_events(tracking_events, save_json=True)

# Use the dict directly in your backend logic
total_people = output["people"]["total"]
recommendations = output["recommendations"]
```

### Option B — Polling JSON file (decoupled / separate process)
```python
import json, time

while True:
    with open("ai/lina/outputs/json/lina_output.json") as f:
        output = json.load(f)
    # process output...
    time.sleep(5)
```

### Option C — REST API Wrapper (Deepika implements)
Deepika's backend should expose an endpoint that:
1. Calls `LinaPipeline.process_events(events)` internally
2. Returns the unified JSON to the frontend

> **Lina does NOT implement any HTTP routes or API endpoints.**
> That is entirely Deepika's responsibility.

---

## 4. What Lina Does NOT Do

| Responsibility | Owner |
|---|---|
| Person detection (YOLO) | Kanimozhi |
| Person tracking (ByteTrack) | Kanimozhi |
| Shelf / inventory detection | Janavi |
| RFID event processing | Janavi |
| REST API endpoints | Deepika |
| Database storage (PostgreSQL) | Deepika |
| Dashboard / frontend | Deepika / Frontend team |
| Hardware / camera simulation | Hardware team |

---

## 5. Configuration Shared With Deepika

Deepika may need to read zone and counter IDs from:

```
ai/lina/config/zones.json
```

Zone IDs (e.g. `ZONE_A`, `BILLING`) and counter IDs (e.g. `COUNTER_1`) in the
JSON output **always match** the `id` fields in `zones.json`.

If new zones are added, only `config/zones.json` needs updating.
No Python code changes are required.

---

## 6. Schema Versioning

The output JSON includes:
```json
"schema_version": "1.0.0"
```

Deepika's backend should check this version field.
If Lina increments the version (e.g. to `"1.1.0"`), it means a breaking schema change
has been made. Both sides must update together.

---

## 7. Error Conditions

If Lina's pipeline receives no valid events, it returns an **empty dict `{}`**.
Deepika's backend should handle this gracefully (e.g. skip the update cycle).

If the prediction model has not been trained yet:
```json
"prediction": {
    "model_available": false,
    "next_15_min": null,
    "next_30_min": null,
    "next_60_min": null,
    "note": "Train the model first using scripts/train_prediction_model.py"
}
```

---

## 8. Contact

For any integration questions regarding **Lina's module**, refer to this document first.
Do not modify files inside `ai/lina/` without coordinating with Lina.
