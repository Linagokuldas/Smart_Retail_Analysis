"""
Waiting Time Tracker.

Tracks individual persons entering and leaving queue regions.
Handles the grace period — a person who briefly disappears from a frame
is NOT assumed to have left the queue immediately.

Privacy: Only anonymous track_ids are stored. No face or identity data.
"""

from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WaitingTimeTracker:
    """
    Tracks entry/exit times for persons in queue regions.
    Maintains per-queue statistics.
    """

    def __init__(self, thresholds: dict):
        """
        Args:
            thresholds: Loaded thresholds.json dict.
        """
        queue_cfg = thresholds.get("queue", {})
        self.grace_period_seconds = float(queue_cfg.get("grace_period_seconds", 10))

        # State: {queue_region_id: {track_id: {entry_time, last_seen, active}}}
        self._active: dict[str, dict[int, dict]] = {}

        # Completed wait records: {queue_region_id: [waiting_time_seconds, ...]}
        self._completed: dict[str, list[float]] = {}

    def update(self, queue_detections: dict[str, dict], current_timestamp: str) -> None:
        """
        Update waiting state from current frame's queue detections.

        Args:
            queue_detections: Output of QueueDetector.detect().
            current_timestamp: ISO 8601 timestamp string for this frame.
        """
        try:
            current_time = datetime.fromisoformat(current_timestamp)
        except ValueError:
            logger.warning("Invalid timestamp format for waiting time update: %s", current_timestamp)
            return

        for region_id, detection in queue_detections.items():
            currently_in = set(detection.get("track_ids", []))

            if region_id not in self._active:
                self._active[region_id] = {}
            if region_id not in self._completed:
                self._completed[region_id] = []

            active = self._active[region_id]
            completed = self._completed[region_id]

            # Mark current in-queue persons as seen
            for track_id in currently_in:
                if track_id not in active:
                    active[track_id] = {
                        "entry_time": current_time,
                        "last_seen": current_time,
                        "active": True,
                    }
                    logger.debug("Queue %s: track_id %s entered at %s", region_id, track_id, current_time)
                else:
                    active[track_id]["last_seen"] = current_time
                    active[track_id]["active"] = True

            # Check persons not seen in current frame
            tracks_to_remove = []
            for track_id, state in active.items():
                if track_id not in currently_in:
                    elapsed = (current_time - state["last_seen"]).total_seconds()
                    if elapsed > self.grace_period_seconds:
                        # Record wait time (can be 0 for single-frame appearances)
                        wait = (state["last_seen"] - state["entry_time"]).total_seconds()
                        completed.append(max(0.0, wait))
                        tracks_to_remove.append(track_id)
                        logger.debug(
                            "Queue %s: track_id %s exited, waited %.1fs",
                            region_id, track_id, wait,
                        )

            for track_id in tracks_to_remove:
                del active[track_id]

    def get_statistics(self, queue_detections: dict[str, dict]) -> dict[str, dict]:
        """
        Compute current waiting time statistics per queue region.

        Args:
            queue_detections: Output of QueueDetector.detect() for the current frame.

        Returns:
            Dict mapping region_id → waiting time stats.
        """
        stats = {}

        for region_id in queue_detections:
            completed = self._completed.get(region_id, [])
            active = self._active.get(region_id, {})

            if completed:
                avg_wait = sum(completed) / len(completed)
                median_wait = float(sorted(completed)[len(completed) // 2])
                max_wait = max(completed)
                min_wait = min(completed)
            else:
                avg_wait = median_wait = max_wait = min_wait = 0.0

            stats[region_id] = {
                "avg_waiting_time_seconds": round(avg_wait, 1),
                "median_waiting_time_seconds": round(median_wait, 1),
                "max_waiting_time_seconds": round(max_wait, 1),
                "min_waiting_time_seconds": round(min_wait, 1),
                "completed_waits": len(completed),
                "currently_waiting": len(active),
            }

        return stats

    def reset(self) -> None:
        """Clear all tracking state."""
        self._active.clear()
        self._completed.clear()
