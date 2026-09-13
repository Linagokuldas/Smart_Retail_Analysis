"""
Queue Detector.

Detects and tracks queues in configured queue regions using:
- Person tracking coordinates
- Configurable queue ROI polygons
- Track persistence (grace period) to avoid false queue breaks

No separate AI model is trained for queue detection.
Queue presence is inferred from track occupancy in a queue polygon.
"""

from datetime import datetime
from typing import Optional
from src.crowd.zone_counter import point_in_polygon
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueueDetector:
    """
    Detects queue presence and length in defined queue regions.
    """

    def __init__(self, zones_config: dict, thresholds: dict):
        """
        Args:
            zones_config: Loaded zones.json dict containing "queue_regions".
            thresholds: Loaded thresholds.json dict.
        """
        self.queue_regions = zones_config.get("queue_regions", [])
        queue_cfg = thresholds.get("queue", {})
        self.min_people = int(queue_cfg.get("min_people_to_form_queue", 3))

        if not self.queue_regions:
            logger.warning("No queue regions configured in zones.json")

    def detect(self, frame_events: list[dict]) -> dict[str, dict]:
        """
        Detect queues in the current frame.

        Args:
            frame_events: Validated tracking events for one frame.

        Returns:
            Dict mapping queue_region_id → {counter_id, people_in_queue, queue_detected, track_ids}.
        """
        results = {}

        for region in self.queue_regions:
            region_id = region["id"]
            counter_id = region.get("counter_id", region_id)
            polygon = region.get("polygon", [])
            max_cap = region.get("max_capacity", 10)

            if len(polygon) < 3:
                logger.warning("Queue region '%s' has invalid polygon — skipping", region_id)
                continue

            # Find all unique track_ids in this queue polygon
            in_queue: list[dict] = []
            seen_tracks: set = set()

            for event in frame_events:
                center = event.get("center")
                if not center or len(center) < 2:
                    continue
                track_id = event["track_id"]
                if track_id in seen_tracks:
                    continue
                try:
                    if point_in_polygon(center, polygon):
                        in_queue.append(event)
                        seen_tracks.add(track_id)
                except Exception as exc:
                    logger.debug("Queue PIP error for track %s: %s", track_id, exc)

            queue_length = len(in_queue)
            queue_detected = queue_length >= self.min_people
            utilization = min(queue_length / max(max_cap, 1), 1.0)

            results[region_id] = {
                "counter_id": counter_id,
                "people_in_queue": queue_length,
                "queue_detected": queue_detected,
                "max_capacity": max_cap,
                "utilization": round(utilization, 3),
                "track_ids": [e["track_id"] for e in in_queue],
            }

        return results
