"""
Validators for the Lina Analytics Module.

Validates incoming tracking event dictionaries before they enter the pipeline.
Invalid events are logged and skipped — they do NOT crash the pipeline.
"""

from typing import Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Required fields in every tracking event
REQUIRED_FIELDS = {"timestamp", "camera_id", "track_id", "class", "confidence", "bbox", "center"}


def validate_tracking_event(event: Any) -> bool:
    """
    Validate a single person tracking event.

    Args:
        event: The raw event dict received from upstream (e.g., Kanimozhi's CV output).

    Returns:
        True if event is valid and can be processed; False otherwise.
    """
    if not isinstance(event, dict):
        logger.warning("Validation failed: event is not a dict — got %s", type(event).__name__)
        return False

    # Check all required fields are present
    missing = REQUIRED_FIELDS - event.keys()
    if missing:
        logger.warning("Validation failed: missing fields %s in event %s", missing, event.get("track_id", "?"))
        return False

    # Validate track_id
    if not isinstance(event["track_id"], (int, str)):
        logger.warning("Validation failed: track_id must be int or str, got %s", type(event["track_id"]))
        return False

    # Validate confidence
    try:
        conf = float(event["confidence"])
        if not (0.0 <= conf <= 1.0):
            logger.warning("Validation failed: confidence %.3f out of [0, 1] range", conf)
            return False
    except (TypeError, ValueError):
        logger.warning("Validation failed: confidence is not numeric — %s", event.get("confidence"))
        return False

    # Validate bbox: must be list/tuple of 4 numbers
    bbox = event.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        logger.warning("Validation failed: bbox must be a list of 4 numbers — got %s", bbox)
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            logger.warning("Validation failed: bbox has zero or negative dimensions [%.1f,%.1f,%.1f,%.1f]", x1, y1, x2, y2)
            return False
    except (TypeError, ValueError):
        logger.warning("Validation failed: bbox contains non-numeric values — %s", bbox)
        return False

    # Validate center: must be list/tuple of 2 numbers
    center = event.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        logger.warning("Validation failed: center must be [cx, cy] — got %s", center)
        return False
    try:
        _ = [float(v) for v in center]
    except (TypeError, ValueError):
        logger.warning("Validation failed: center contains non-numeric values — %s", center)
        return False

    # Validate timestamp is a non-empty string
    ts = event.get("timestamp", "")
    if not isinstance(ts, str) or not ts.strip():
        logger.warning("Validation failed: timestamp must be a non-empty string — got %s", ts)
        return False

    return True


def validate_events_batch(events: list[Any]) -> list[dict]:
    """
    Validate a list of tracking events, filtering out invalid ones.

    Args:
        events: List of raw tracking event dicts.

    Returns:
        List containing only valid events.
    """
    if not isinstance(events, list):
        logger.error("Expected list of events, got %s", type(events).__name__)
        return []

    valid = [e for e in events if validate_tracking_event(e)]
    dropped = len(events) - len(valid)
    if dropped > 0:
        logger.warning("Dropped %d invalid events out of %d total", dropped, len(events))

    return valid


def validate_zone_config(zones_config: dict) -> bool:
    """
    Validate the zones configuration dictionary.

    Args:
        zones_config: Loaded zones.json as a dict.

    Returns:
        True if configuration is valid.
    """
    if "zones" not in zones_config:
        logger.error("Zone config missing 'zones' key")
        return False

    for zone in zones_config["zones"]:
        for field in ("id", "name", "polygon", "area_sqm"):
            if field not in zone:
                logger.error("Zone missing required field '%s': %s", field, zone.get("id", "?"))
                return False
        polygon = zone["polygon"]
        if not isinstance(polygon, list) or len(polygon) < 3:
            logger.error("Zone '%s': polygon must have at least 3 points", zone.get("id"))
            return False

    return True
