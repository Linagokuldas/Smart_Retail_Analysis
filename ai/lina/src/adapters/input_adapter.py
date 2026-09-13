"""
Input Adapter for the Lina Analytics Module.

Responsible for:
- Accepting tracking events from upstream (Kanimozhi's CV pipeline or mock data)
- Validating and normalizing them into the standard internal format
- Grouping events by timestamp frame for batch processing

--- INTEGRATION CONTRACT (for Kanimozhi) ---
Upstream must produce a list of dicts in this format:
{
    "timestamp": "2026-09-13T12:30:15",   # ISO 8601 string
    "camera_id": "CAM01",                  # Camera identifier string
    "track_id": 27,                        # Unique integer track ID
    "class": "person",                     # Object class string
    "confidence": 0.94,                    # Float in [0.0, 1.0]
    "bbox": [x1, y1, x2, y2],             # Pixel coords (floats or ints)
    "center": [cx, cy]                     # Centroid pixel coords
}

If upstream is not ready, use data/sample/people_tracks.json as mock input.
---
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from src.utils.validators import validate_events_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InputAdapter:
    """
    Normalizes and validates incoming person tracking events.

    Can ingest:
    1. A Python list of raw dicts (real-time mode)
    2. A JSON file path (batch/demo mode)
    """

    def __init__(self, min_confidence: float = 0.50):
        """
        Args:
            min_confidence: Minimum confidence score to accept an event.
        """
        self.min_confidence = min_confidence

    def from_list(self, raw_events: list[Any]) -> list[dict]:
        """
        Process a list of raw tracking event dicts.

        Args:
            raw_events: List of raw person tracking events.

        Returns:
            List of validated, normalized events.
        """
        valid_events = validate_events_batch(raw_events)
        return self._normalize(valid_events)

    def from_json_file(self, filepath: str | Path) -> list[dict]:
        """
        Load and process tracking events from a JSON file.

        Args:
            filepath: Path to JSON file containing a list of tracking events.

        Returns:
            List of validated, normalized events.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error("Input JSON file not found: %s", filepath)
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load input JSON: %s", exc)
            return []

        if not isinstance(raw, list):
            logger.error("Input JSON must be a list of events; got %s", type(raw).__name__)
            return []

        logger.info("Loaded %d raw events from %s", len(raw), filepath.name)
        return self.from_list(raw)

    def group_by_frame(self, events: list[dict]) -> dict[str, list[dict]]:
        """
        Group validated events by their timestamp (treating each unique
        timestamp as one 'frame').

        Args:
            events: List of validated tracking events.

        Returns:
            OrderedDict mapping timestamp string → list of events in that frame.
        """
        frames: dict[str, list[dict]] = {}
        for event in events:
            ts = event["timestamp"]
            frames.setdefault(ts, []).append(event)

        # Sort frames chronologically
        sorted_frames = dict(sorted(frames.items()))
        logger.info("Grouped events into %d frames", len(sorted_frames))
        return sorted_frames

    def _normalize(self, events: list[dict]) -> list[dict]:
        """
        Normalize validated events to ensure consistent types.

        Args:
            events: Validated event dicts.

        Returns:
            Normalized event dicts.
        """
        normalized = []
        for e in events:
            try:
                conf = float(e["confidence"])
                if conf < self.min_confidence:
                    logger.debug(
                        "Skipping track_id=%s: confidence %.2f < %.2f",
                        e["track_id"], conf, self.min_confidence,
                    )
                    continue

                normalized.append({
                    "timestamp": e["timestamp"].strip(),
                    "camera_id": str(e["camera_id"]).strip(),
                    "track_id": int(e["track_id"]),
                    "class": str(e.get("class", "person")).lower().strip(),
                    "confidence": conf,
                    "bbox": [float(v) for v in e["bbox"]],
                    "center": [float(v) for v in e["center"]],
                })
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Normalization failed for event: %s | error: %s", e, exc)

        logger.info("Normalized %d events (filtered by min_confidence=%.2f)", len(normalized), self.min_confidence)
        return normalized
